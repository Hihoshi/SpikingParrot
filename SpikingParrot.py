import torch
import torch.nn as nn
import snntorch as snn
from MyLayer import Dyt, surrogate_dyt
import random


class SpikingLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        dropout: float
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.mixer = nn.Linear(hidden_dim + input_dim, hidden_dim)

        self.norm = Dyt(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.slstm = snn.SLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            spike_grad=surrogate_dyt(hidden_dim),
            learn_threshold=True,
            init_hidden=False,
        )
        self.syn = None
        self.mem = None
        self.initialized = False

    def init_hidden(
        self, batch_size: int, device: torch.device,
        syn_init: torch.Tensor = None, mem_init: torch.Tensor = None
    ) -> None:
        syn = torch.zeros(batch_size, self.hidden_dim, device=device) \
            if syn_init is None else syn_init
        mem = torch.zeros(batch_size, self.hidden_dim, device=device) \
            if mem_init is None else mem_init
        self.syn = syn
        self.mem = mem
        self.initialized = True

    def _forward_step(self, x: torch.Tensor) -> torch.Tensor:
        spk, self.syn, self.mem = self.slstm(x, self.syn, self.mem)
        output = self.linear(spk)
        output = self.mixer(torch.cat((output, x), dim=-1))

        return self.dropout(self.norm(output)), self.syn, self.mem

    def _forward_seq(self, x: torch.Tensor) -> torch.Tensor:

        _, S, _ = x.shape
        spks, syns, mems = [], [], []
        for step in range(S):
            spk, self.syn, self.mem = self.slstm(x[:, step, :], self.syn, self.mem)
            spks.append(spk)
            syns.append(self.syn)
            mems.append(self.mem)

        spks = torch.stack(spks, dim=1)
        syns = torch.stack(syns, dim=1)
        mems = torch.stack(mems, dim=1)

        outputs = self.linear(spks)
        outputs = self.mixer(torch.cat((outputs, x), dim=-1))

        return self.dropout(self.norm(outputs)), syns, mems

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.initialized is False:
            self.init_hidden(x.size(0), x.device)
        if x.dim() == 3:
            return self._forward_seq(x)
        elif x.dim() == 2:
            return self._forward_step(x)
        else:
            raise ValueError("input dim must be 2(step) or 3(sequence)")

    def forward_step(self, x: torch.Tensor, syn: torch.Tensor, mem: torch.Tensor) -> tuple:
        spk, new_syn, new_mem = self.slstm(x, syn, mem)
        output = self.linear(spk)
        output = self.mixer(torch.cat((output, x), dim=-1))
        output = self.dropout(self.norm(output))
        return output, new_syn, new_mem

    def reset(self) -> None:
        self.syn = None
        self.mem = None
        self.initialized = False


class SpikingEncoder(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.norm = Dyt(embedding_dim)

        self.layers = nn.ModuleList([
            SpikingLayer(input_dim=hidden_dim * 2 if i == 0 else hidden_dim, hidden_dim=hidden_dim, dropout=dropout) \
            for i in range(num_layers)
        ])
        self.bislstm = nn.ModuleDict({
            "f": SpikingLayer(input_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout),
            "b": SpikingLayer(input_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
        })

    def forward(self, src: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        src = self.norm(self.embedding(src))
        f_output, _, _ = self.bislstm["f"](src)
        b_output, _, _ = self.bislstm["b"](torch.flip(src, dims=[1]))
        output = torch.cat((f_output, torch.flip(b_output, dims=[1])), dim=-1)

        syns, mems = [], []
        for _, layer in enumerate(self.layers):
            output, syn, mem = layer(output)
            syns.append(syn)
            mems.append(mem)
        return syns, mems

    def reset(self) -> None:
        self.bislstm["f"].reset()
        self.bislstm["b"].reset()
        for layer in self.layers:
            layer.reset()


# general Luong attention
class Attention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.w.weight)

        self.norm = Dyt(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_mem, encoder_mems):
        """
        decoder_mem: (batch, hidden_dim)
        encoder_mems: (batch, seq_len, hidden_dim)
        """
        decoder_mem = self.w(decoder_mem).unsqueeze(2)                  # (batch, hidden_dim, 1)

        # compute scores = batch matmul
        scores = torch.bmm(encoder_mems, decoder_mem).squeeze(-1)       # (batch, seq_len)

        # softmax over encoder seq_len
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)      # (batch, seq_len)

        # weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_mems)    # (batch, 1, hidden_dim)
        context = context.squeeze(1)                                    # (batch, hidden_dim)

        return self.dropout(self.norm(context))


class SpikingDecoder(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.initialized = False

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.norm = Dyt(embedding_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": Attention(hidden_dim, dropout),
                "spiking": SpikingLayer(
                    input_dim=embedding_dim if i == 0 else hidden_dim * 2,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            }) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def init_hidden(
        self, batch_size: int, device: torch.device,
        encoder_syns: list[torch.Tensor], encoder_mems: list[torch.Tensor]
    ) -> None:
        for i, layer in enumerate(self.layers):
            layer["spiking"].init_hidden(
                batch_size, device,
                encoder_syns[i][:, -1, :],
                encoder_mems[i][:, -1, :],
            )
        self.initialized = True

    def _forward_step(
        self, x: torch.Tensor,
        encoder_mems: list[torch.Tensor],
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            output, _, _ = layer["spiking"](x)
            mem = layer["spiking"].mem
            context = layer["attention"](mem, encoder_mems[i])
            x = torch.cat((output, context), dim=-1)
        return x

    def _forward_seq(
        self, tgt: torch.Tensor,
        encoder_mems: list[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        _, S, _ = tgt.shape
        outputs = []
        for step in range(S):
            if step == 0 or random.random() < teacher_forcing_ratio:
                outputs.append(self._forward_step(tgt[:, step, :], encoder_mems))
            else:
                logits = torch.argmax(self.fc(outputs[-1].clone().detach()), dim=-1)
                previous = self.norm(self.embedding(logits))
                outputs.append(self._forward_step(previous, encoder_mems))
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward(
        self, tgt: torch.Tensor,
        encoder_syns: list[torch.Tensor],
        encoder_mems: list[torch.Tensor],
        teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        B = tgt.size(0)
        device = tgt.device
        if self.initialized is False:
            self.init_hidden(B, device, encoder_syns, encoder_mems)
        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self.fc(self._forward_seq(tgt, encoder_mems, teacher_forcing_ratio))
        elif tgt.dim() == 2:
            return self.fc(self._forward_step(tgt, encoder_mems))
        else:
            raise ValueError("input dim must be 2(step) or 3(sequence)")

    def step(
        self, input_token: torch.Tensor,
        hidden_states: list,
        encoder_mems: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list]:
        x = self.embedding(input_token)
        x = self.norm(x)
        
        new_hidden_states = []
        for i, layer_dict in enumerate(self.layers):
            syn, mem = hidden_states[i]
            output, new_syn, new_mem = layer_dict["spiking"].forward_step(x, syn, mem)
            context = layer_dict["attention"](new_mem, encoder_mems[i])
            x = torch.cat((output, context), dim=-1)
            new_hidden_states.append((new_syn, new_mem))
        
        logits = self.fc(x)
        return logits, new_hidden_states

    def reset(self) -> None:
        for layer in self.layers:
            layer["spiking"].reset()
        self.initialized = False


class SpikingParrot(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.encoder = SpikingEncoder(
            padding_idx, embedding_dim, vocab_size,
            hidden_dim, num_layers,
            dropout,
        )
        self.decoder = SpikingDecoder(
            padding_idx, embedding_dim, vocab_size,
            hidden_dim, num_layers,
            dropout,
        )

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        self.reset()
        encoder_syns, encoder_mems = self.encoder(src)
        outputs = self.decoder(
            tgt, encoder_syns, encoder_mems,
            teacher_forcing_ratio
        )
        return outputs

    def reset(self) -> None:
        self.encoder.reset()
        self.decoder.reset()

    def greedy_decode(
        self, src: torch.Tensor,
        bos_token_id: int, eos_token_id: int, max_length: int
    ) -> list:
        B, _ = src.shape
        device = src.device
        self.reset()

        with torch.no_grad():
            encoder_syns, encoder_mems = self.encoder(src)

            input_ids = torch.full(
                (B, 1), bos_token_id,
                dtype=torch.long, device=device
            )

            for _ in range(max_length):
                current_input = input_ids[:, -1]
                output = self.decoder(
                    current_input,
                    encoder_syns, encoder_mems
                )
                next_tokens = torch.argmax(output, dim=-1).unsqueeze(1)
                input_ids = torch.cat([input_ids, next_tokens], dim=1)

        sequences = []
        for seq in input_ids:
            seq_list = seq.tolist()
            try:
                eos_pos = seq_list.index(eos_token_id)
                seq_list = seq_list[: eos_pos + 1]
            except ValueError:
                pass
            sequences.append(seq_list)

        return sequences

    def beam_decode(
        self, src: torch.Tensor,
        bos_token_id: int, eos_token_id: int,
        max_length: int, beam_size: int = 4,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.1
    ) -> list[list[int]]:
        self.reset()
        batch_size = src.size(0)
        device = src.device

        with torch.no_grad():
            encoder_syns, encoder_mems = self.encoder(src)
            
            self.decoder.init_hidden(batch_size, device, encoder_syns, encoder_mems)
            
            init_hidden_states = []
            for layer in self.decoder.layers:
                init_hidden_states.append((layer["spiking"].syn, layer["spiking"].mem))
            
            expanded_encoder_mems = []
            for mem in encoder_mems:
                # mem shape: (batch, seq_len, hidden_dim)
                expanded_mem = mem.unsqueeze(1).repeat(1, beam_size, 1, 1)
                expanded_mem = expanded_mem.view(batch_size * beam_size, mem.size(1), mem.size(2))
                expanded_encoder_mems.append(expanded_mem)
            
            input_ids = torch.full(
                (batch_size * beam_size, 1), bos_token_id,
                dtype=torch.long, device=device
            )
            
            generated_histories = torch.zeros(
                batch_size * beam_size, self.decoder.vocab_size,
                dtype=torch.float, device=device
            )
            
            # init hidden states
            hidden_states = []
            for syn, mem in init_hidden_states:
                expanded_syn = syn.unsqueeze(1).repeat(1, beam_size, 1)
                expanded_mem = mem.unsqueeze(1).repeat(1, beam_size, 1)
                hidden_states.append((expanded_syn, expanded_mem))
            
            beam_scores = torch.full((batch_size, beam_size), -float('inf'), device=device)
            beam_scores[:, 0] = 0.0
            beam_scores = beam_scores.view(-1)  # (batch_size * beam_size,)
            
            beam_lengths = torch.ones(batch_size * beam_size, dtype=torch.long, device=device)
            
            finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
            
            for step in range(1, max_length + 1):
                current_input = input_ids[:, -1]
                
                if repetition_penalty != 1.0 and step > 1:
                    for i in range(batch_size * beam_size):
                        if not finished[i]:
                            generated_histories[i, current_input[i]] += 1.0
                
                flat_hidden_states = []
                for syn, mem in hidden_states:
                    flat_syn = syn.view(-1, syn.size(-1))
                    flat_mem = mem.view(-1, mem.size(-1))
                    flat_hidden_states.append((flat_syn, flat_mem))
                
                logits, new_flat_hidden_states = self.decoder.step(
                    current_input, flat_hidden_states, expanded_encoder_mems
                )
                
                new_hidden_states = []
                for i, (syn, mem) in enumerate(new_flat_hidden_states):
                    syn = syn.view(batch_size, beam_size, -1)
                    mem = mem.view(batch_size, beam_size, -1)
                    new_hidden_states.append((syn, mem))
                
                if repetition_penalty != 1.0:
                    penalty_factors = repetition_penalty ** generated_histories
                    logits = logits / penalty_factors
                
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                vocab_size = log_probs.size(-1)
                
                if finished.any():
                    log_probs[finished] = -float('inf')
                    log_probs[finished, eos_token_id] = 0.0
                
                candidate_scores = beam_scores.unsqueeze(1) + log_probs  # (batch*beam, vocab)
                
                if length_penalty != 0.0:
                    length_penalties = ((beam_lengths + 1).float() ** length_penalty).unsqueeze(1)
                    candidate_scores = candidate_scores / length_penalties
                
                candidate_scores = candidate_scores.view(batch_size, beam_size * vocab_size)
                
                top_scores, top_indices = candidate_scores.topk(beam_size, dim=1)
                
                beam_indices = top_indices // vocab_size  # (batch, beam)
                token_indices = top_indices % vocab_size  # (batch, beam)
                
                raw_candidate_scores = beam_scores.unsqueeze(1) + log_probs
                raw_candidate_scores = raw_candidate_scores.view(batch_size, beam_size * vocab_size)
                new_beam_scores = torch.gather(raw_candidate_scores, 1, top_indices)
                beam_scores = new_beam_scores.view(-1)
                
                prev_input_ids = input_ids.view(batch_size, beam_size, -1)
                selected_sequences = torch.gather(
                    prev_input_ids, 1,
                    beam_indices.unsqueeze(-1).expand(-1, -1, prev_input_ids.size(-1))
                )
                new_tokens = token_indices.unsqueeze(-1)
                input_ids = torch.cat([selected_sequences, new_tokens], dim=-1)
                input_ids = input_ids.view(batch_size * beam_size, -1)
                
                prev_lengths = beam_lengths.view(batch_size, beam_size)
                selected_lengths = torch.gather(prev_lengths, 1, beam_indices)
                
                prev_finished = finished.view(batch_size, beam_size)
                selected_finished = torch.gather(prev_finished, 1, beam_indices)
                new_finished = selected_finished | (token_indices == eos_token_id)
                finished = new_finished.view(-1)
                
                length_increment = (~new_finished).long()
                beam_lengths = (selected_lengths + length_increment).view(-1)
                
                updated_hidden_states = []
                for syn, mem in new_hidden_states:
                    # syn, mem shape: (batch, beam, hidden)
                    beam_indices_expanded = beam_indices.unsqueeze(-1).expand(-1, -1, syn.size(-1))
                    selected_syn = torch.gather(syn, 1, beam_indices_expanded)
                    selected_mem = torch.gather(mem, 1, beam_indices_expanded)
                    updated_hidden_states.append((selected_syn, selected_mem))
                hidden_states = updated_hidden_states
                
                if repetition_penalty != 1.0:
                    prev_histories = generated_histories.view(batch_size, beam_size, -1)
                    beam_indices_expanded = beam_indices.unsqueeze(-1).expand(-1, -1, prev_histories.size(-1))
                    generated_histories = torch.gather(prev_histories, 1, beam_indices_expanded)
                    generated_histories = generated_histories.view(batch_size * beam_size, -1)
                
                if finished.all():
                    break
            
            if length_penalty != 0.0:
                final_scores = beam_scores / (beam_lengths.float() ** length_penalty)
            else:
                final_scores = beam_scores
            
            final_scores = final_scores.view(batch_size, beam_size)
            best_beam_indices = final_scores.argmax(dim=1)

            input_ids = input_ids.view(batch_size, beam_size, -1)
            best_sequences = []
            
            for i in range(batch_size):
                seq = input_ids[i, best_beam_indices[i]].tolist()
                if eos_token_id in seq:
                    eos_pos = seq.index(eos_token_id)
                    seq = seq[:eos_pos + 1]
                best_sequences.append(seq)
            
            return best_sequences

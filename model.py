import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as utils
from normalizer import Dyt
from binarizer import sdyt

class SpikingLayer(nn.Module):
    def __init__(self,
        input_dim: int, hidden_dim: int,
        residual: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.normalizer = Dyt(hidden_dim)
        self.slstm = snn.SLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            spike_grad=sdyt(hidden_dim),
            learn_threshold=True,
            init_hidden=False,
        )

    def forward(self, x, syn=None, mem=None):
        """
        1. sequence mode (train):  x.shape = (B, S, D)
        2. step mode (inference):  x.shape = (B, D)
        """
        if x.dim() == 3:
            return self._forward_seq(x, syn, mem)
        elif x.dim() == 2:
            return self._forward_step(x, syn, mem)
        else:
            ValueError("input dimension should be 2 or 3 (single step or a whole sequence)")

    def _forward_seq(self, x, syn=None, mem=None):
        """process a whole sequence once (batch, seq_len, input_dim)"""
        B, S, _ = x.shape
        # preallocate space of variables
        spks = torch.zeros(B, S, self.hidden_dim, device=x.device)

        # 初始化隐藏状态
        syn = torch.zeros(B, self.hidden_dim, device=x.device) \
            if syn is None else syn
        mem = torch.zeros(B, self.hidden_dim, device=x.device) \
            if mem is None else mem

        for step in range(S):
            spk, syn, mem = self.slstm(x[:, step, :], syn, mem)
            spks[:, step, :] = spk

        if self.residual:
            outputs = self.normalizer(self.fc(spks) + x)
        else:
            outputs = self.normalizer(self.fc(spks))

        return outputs, syn, mem

    def _forward_step(self, x, syn, mem):
        """process one time step at a time (batch, input_dim)"""
        B, _ = x.shape
        syn = torch.zeros(B, self.hidden_dim, device=x.device) \
            if syn is None else syn
        mem = torch.zeros(B, self.hidden_dim, device=x.device) \
            if mem is None else mem

        spk, syn, mem = self.slstm(x, syn, mem)
        output = self.fc(spk)
        if self.residual:
            output = self.normalizer(output + x)
        else:
            output = self.normalizer(output)
        return output, syn, mem


class SpikingEncoder(nn.Module):
    def __init__(
        self, padding_idx : int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional:bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList()

        # the fisrt input dim is embedding dim
        input_dim = embedding_dim
        for i in range(num_layers):
            if bidirectional:
                self.layers.append(
                    nn.ModuleDict({
                        "f_layer": SpikingLayer(input_dim, hidden_dim, residual=(i > 0)),
                        "b_layer": SpikingLayer(input_dim, hidden_dim, residual=(i > 0)),
                    })
                )
            else:
                self.layers.append(SpikingLayer(input_dim, hidden_dim, residual=(i > 0)))
            # follow up layers's input dim changes to hidden dim
            input_dim = hidden_dim

    def forward(self, src):
        src = self.norm(self.embedding(src))
        f_outputs = src
        b_outputs = torch.flip(src, dims=[1])
        syns, mems = [], []
        for _, layer  in enumerate(self.layers):
            if self.bidirectional:
                f_outputs, f_syn, f_mem = layer["f_layer"](f_outputs)
                b_outputs, b_syn, b_mem = layer["b_layer"](b_outputs)
            else:
                outputs, syn, mem = layer(src)
            # output aligned syn, mem of the last layer
            if self.bidirectional:
                syn = torch.cat([f_syn, b_syn], dim=-1)
                mem = torch.cat([f_mem, b_mem], dim=-1)
            # append each layer's syn, mem to the list
            syns.append(syn)
            mems.append(mem)
        return syns, mems


class SpikingDecoder(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList()

        input_dim = embedding_dim
        for i in range(num_layers):
            self.layers.append(SpikingLayer(input_dim, hidden_dim, residual=(i > 0)))
            input_dim = hidden_dim

    def forward(self, tgt, syn, mem):
        """
        1. sequence mode (train):  x.shape = (B, S, D)
        2. step mode (inference):  x.shape = (B, D)
        """
        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self._forward_seq(tgt, syn, mem)
        elif tgt.dim() == 2:
            return self._forward_step(tgt, syn, mem)
        else:
            raise ValueError("input dimension should be 2 or 3 (single step or a whole sequence)")

    def _forward_seq(self, tgt, syns, mems):
        for i, layer in enumerate(self.layers):
            outputs, _, _ = layer._forward_seq(tgt, syns[i], mems[i]) 
            tgt = outputs
        return outputs

    def _forward_step(self, tgt, syns, mems):
        for i, layer in enumerate(self.layers):
            outputs, syns[i], mems[i] = layer._forward_step(tgt, syns[i], mems[i])
            tgt = outputs

        return outputs, syns, mems


class SpikingParrot(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = SpikingEncoder(padding_idx, embedding_dim, vocab_size, hidden_dim, num_layers, bidirectional)
        if bidirectional:
            self.decoder = SpikingDecoder(padding_idx, embedding_dim, vocab_size, hidden_dim * 2, num_layers)
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        else:
            self.decoder = SpikingDecoder(padding_idx, embedding_dim, vocab_size, hidden_dim, num_layers)
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        syn, mem = self.encoder(src)
        outputs = self.decoder(tgt, syn, mem)
        return self.fc(outputs)

    def greedy_decode(self, src, bos_token_id, eos_token_id, max_length=32):
        batch_size, _ = src.shape
        device = src.device
        syns, mems = self.encoder(src)
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        for _ in range(max_length - 1):
            current_input = input_ids[:, -1]
            embedded = self.decoder.embedding(current_input)
            embedded = self.decoder.norm(embedded)
            output, syns, mems = self.decoder._forward_step(embedded, syns, mems)
            logits = self.fc(output)
            next_tokens = torch.argmax(torch.log_softmax(logits, dim=-1), dim=-1).unsqueeze(1)
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

    def beam_search(self, src, bos_token_id, eos_token_id, width=4, max_length=32):
        device = src.device
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search supports batch size 1 only"
        syns, mems = self.encoder(src)
        initial_seq = torch.tensor([[bos_token_id]], device=device)
        initial_score = 0.0
        beams = [(initial_seq, initial_score, [s.clone() for s in syns], [m.clone() for m in mems])]
        completed = []
        for _ in range(max_length):
            new_beams = []
            for seq, score, syns_state, mems_state in beams:
                if seq[0, -1] == eos_token_id:
                    completed.append((seq, score))
                    continue
                current_input = seq[:, -1]
                embedded = self.decoder.embedding(current_input)
                embedded = self.decoder.norm(embedded)
                output, new_syns, new_mems = self.decoder._forward_step(embedded, syns_state, mems_state)
                logits = self.fc(output)
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_probs, topk_indices = log_probs.topk(width, dim=-1)
                for i in range(width):
                    token = topk_indices[0, i].item()
                    new_score = score + topk_probs[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([[token]], device=device)], dim=1)
                    new_syns_clone = [s.clone() for s in new_syns]
                    new_mems_clone = [m.clone() for m in new_mems]
                    new_beams.append((new_seq, new_score, new_syns_clone, new_mems_clone))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:width]
            beams = [b for b in beams if b[0][0, -1] != eos_token_id]
            if not beams:
                break
        completed += [(b[0], b[1]) for b in beams]
        completed.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0][0].tolist() if completed else []
        try:
            eos_pos = best_seq.index(eos_token_id)
            best_seq = best_seq[: eos_pos + 1]
        except ValueError:
            pass
        return [best_seq]

    def reset(self):
        for i in range(self.num_layers):
            if self.bidirectional:
                utils.reset(self.encoder.layers[i]["f_layer"].slstm)
                utils.reset(self.encoder.layers[i]["b_layer"].slstm)
            else:
                utils.reset(self.encoder.layers[i].slstm)
            utils.reset(self.decoder.layers[i].slstm)

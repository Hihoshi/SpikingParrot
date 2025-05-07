import torch
import torch.nn as nn
import snntorch as snn
from mylayer import Dyt, surrogate_dyt
import random


class SpikingLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        dropout: float, mix: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mix = mix

        self.linear = nn.Linear(hidden_dim, hidden_dim)
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

        if mix:
            self.mixer = nn.Linear(hidden_dim + input_dim, hidden_dim)

    def init_hidden(
        self, batch_size: int, device: torch.device,
        syn_init: torch.tensor = None, mem_init: torch.tensor = None,
    ) -> None:
        syn = torch.zeros(batch_size, self.hidden_dim, device=device) \
            if syn_init is None else syn_init
        mem = torch.zeros(batch_size, self.hidden_dim, device=device) \
            if mem_init is None else mem_init
        self.syn = syn
        self.mem = mem
        self.initialized = True

    def _forward_step(self, x: torch.tensor) -> torch.tensor:
        spk, self.syn, self.mem = self.slstm(x, self.syn, self.mem)
        if self.mix:
            output = self.mixer(torch.cat((self.linear(spk), x), dim=-1))
        else:
            output = self.linear(spk)
        return self.dropout(self.norm(output))

    def _forward_seq(self, x: torch.tensor) -> torch.tensor:
        B, S, _ = x.shape
        spks = torch.zeros(B, S, self.hidden_dim, device=x.device)
        for step in range(S):
            spk, self.syn, self.mem = self.slstm(x[:, step, :], self.syn.clone(), self.mem.clone())
            spks[:, step, :] = spk

        if self.mix:
            outputs = self.mixer(torch.cat((self.linear(spks), x), dim=-1))
        else:
            outputs = self.linear(spks)
        return self.dropout(self.norm(outputs))

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.initialized is False:
            self.init_hidden(x.size(0), x.device)
        if x.dim() == 3:
            return self._forward_seq(x)
        elif x.dim() == 2:
            return self._forward_step(x)
        else:
            ValueError("输入维度必须为2(单步)或3(序列)")

    def reset(self) -> None:
        self.syn = None
        self.mem = None
        self.initialized = False


class SpikingEncoder(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList()
        if bidirectional:
            self.syn_fc = nn.Linear(hidden_dim * 2, hidden_dim)
            self.mem_fc = nn.Linear(hidden_dim * 2, hidden_dim)
            self.output_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        for i in range(num_layers):
            if bidirectional:
                self.layers.append(
                    nn.ModuleDict({
                        "f_layer": SpikingLayer(
                            embedding_dim if i == 0 else hidden_dim,
                            hidden_dim, dropout
                        ),
                        "b_layer": SpikingLayer(
                            embedding_dim if i == 0 else hidden_dim,
                            hidden_dim, dropout
                        ),
                    })
                )
            else:
                self.layers.append(
                    SpikingLayer(
                        embedding_dim if i == 0 else hidden_dim,
                        hidden_dim, dropout
                    )
                )

    def forward(self, src: torch.tensor) -> tuple[list[torch.tensor], list[torch.tensor], list[torch.tensor]]:
        src = self.norm(self.embedding(src))
        f_out = src
        b_out = torch.flip(src.clone(), dims=[1]) if self.bidirectional else None
        outputs, syns, mems = [], [], []

        for _, layer in enumerate(self.layers):
            if self.bidirectional:
                f_out = layer["f_layer"](f_out)
                b_out = layer["b_layer"](b_out)
                b_out = torch.flip(b_out, dims=[1])
                f_syn, f_mem = layer["f_layer"].syn, layer["f_layer"].mem
                b_syn, b_mem = layer["b_layer"].syn, layer["b_layer"].mem
                syn = self.syn_fc(torch.cat((f_syn, b_syn), dim=-1))
                mem = self.mem_fc(torch.cat((f_mem, b_mem), dim=-1))
                output = self.output_fc(torch.cat((f_out, b_out), dim=-1))
            else:
                f_out = layer(f_out)
                syn, mem = layer.syn, layer.mem
                output = f_out
            syns.append(syn)
            mems.append(mem)
            outputs.append(output)
        return outputs, syns, mems

    def reset(self) -> None:
        for layer in self.layers:
            if self.bidirectional:
                layer["f_layer"].reset()
                layer["b_layer"].reset()
            else:
                layer.reset()


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = Dyt(hidden_dim)

    def forward(
        self, decoder_hidden_state: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        query = self.q_proj(decoder_hidden_state).unsqueeze(1)
        key = self.k_proj(encoder_outputs)
        value = self.v_proj(encoder_outputs)

        context_vector = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0, is_causal=False
        )

        context_vector = self.norm(context_vector.squeeze(1))
        return context_vector


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

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": Attention(hidden_dim),
                "spiking": SpikingLayer(
                    embedding_dim if i == 0 else hidden_dim * 2,
                    hidden_dim, dropout
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
                encoder_syns[i],
                encoder_mems[i],
            )
        self.initialized = True

    def _forward_step(
        self, tgt: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            output = layer["spiking"](tgt)
            mem = layer["spiking"].mem
            context = layer["attention"](mem, encoder_outputs[i])
            next_tgt = torch.cat((output, context), dim=-1)
            tgt = next_tgt
        return tgt

    def _forward_seq(
        self, tgt: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        B, S, _ = tgt.shape
        device = tgt.device
        outputs = torch.zeros(B, S, self.hidden_dim * 2, device=device)
        for step in range(S):
            if step == 0 or random.random() < teacher_forcing_ratio:
                outputs[:, step, :] = self._forward_step(tgt[:, step, :], encoder_outputs)
            else:
                input_id = torch.argmax(self.fc(outputs[:, step - 1, :]).detach(), dim=-1)
                previous_tgt = self.norm(self.embedding(input_id.detach()))
                outputs[:, step, :] = self._forward_step(previous_tgt, encoder_outputs)
        return self.fc(outputs)

    def forward(
        self, tgt: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
        encoder_syns: list[torch.Tensor],
        encoder_mems: list[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        B = tgt.size(0)
        device = tgt.device
        if self.initialized is False:
            self.init_hidden(B, device, encoder_syns, encoder_mems)
        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self._forward_seq(tgt, encoder_outputs, teacher_forcing_ratio)
        elif tgt.dim() == 2:
            return self.fc(self._forward_step(tgt, encoder_outputs))
        else:
            raise ValueError("输入维度必须为2(单步)或3(序列)")

    def reset(self) -> None:
        for layer in self.layers:
            layer["spiking"].reset()
        self.initialized = False


class SpikingParrot(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = SpikingEncoder(
            padding_idx, embedding_dim, vocab_size,
            hidden_dim, num_layers, bidirectional,
            dropout,
        )
        self.decoder = SpikingDecoder(
            padding_idx, embedding_dim, vocab_size,
            hidden_dim, num_layers,
            dropout,
        )

    def forward(
        self, src: torch.tensor, tgt: torch.tensor,
        teacher_forcing_ratio: float = 0.0
    ) -> torch.tensor:
        self.reset()
        encoder_outputs, encoder_syns, encoder_mems = self.encoder(src)
        outputs = self.decoder(
            tgt, encoder_outputs,
            encoder_syns, encoder_mems,
            teacher_forcing_ratio
        )
        return outputs

    def reset(self) -> None:
        self.encoder.reset()
        self.decoder.reset()

    def greedy_decode(
        self, src: torch.tensor,
        bos_token_id: int, eos_token_id: int, max_length: int
    ) -> list:
        B, _ = src.shape
        device = src.device

        with torch.no_grad():
            encoder_outputs, encoder_syns, encoder_mems = self.encoder(src)

            input_ids = torch.full(
                (B, 1), bos_token_id,
                dtype=torch.long, device=device
            )

            for _ in range(max_length):
                current_input = input_ids[:, -1]
                output = self.decoder(
                    current_input, encoder_outputs,
                    encoder_syns, encoder_mems
                )
                next_tokens = torch.argmax(output, dim=-1)
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

        self.reset()
        return sequences

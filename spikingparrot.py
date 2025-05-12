import torch
import torch.nn as nn
import snntorch as snn
from mylayer import Dyt, surrogate_dyt
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
        syn_init: torch.Tensor = None, mem_init: torch.Tensor = None,
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

        return self.dropout(self.norm(output))

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

    def reset(self) -> None:
        self.syn = None
        self.mem = None
        self.initialized = False


# class SpikingEncoder(nn.Module):
#     def __init__(
#         self, padding_idx: int,
#         embedding_dim: int, vocab_size: int,
#         hidden_dim: int, num_layers: int,
#         dropout: float,
#     ):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
#         self.norm = Dyt(embedding_dim)
#         self.bislstm = nn.ModuleList()

#         for i in range(num_layers):
#             self.bislstm.append(
#                 nn.ModuleDict({
#                     "f": SpikingLayer(embedding_dim if i == 0 else hidden_dim, hidden_dim, dropout),
#                     "b": SpikingLayer(embedding_dim if i == 0 else hidden_dim, hidden_dim, dropout),
#                 })
#             )

#     def forward(self, src: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
#         src = self.norm(self.embedding(src))
#         f_out = src
#         b_out = torch.flip(src.clone(), dims=[1])

#         outputs, syns, mems = [], [], []
#         for _, layer in enumerate(self.bislstm):
#             f_out = layer["f"](f_out)
#             b_out = layer["b"](b_out)
#             f_syn, f_mem = layer["f"].syn, layer["f"].mem
#             b_syn, b_mem = layer["b"].syn, layer["b"].mem

#             syn = torch.cat((f_syn, b_syn), dim=-1)
#             mem = torch.cat((f_mem, b_mem), dim=-1)
#             output = torch.cat((f_out, torch.flip(b_out, dims=[1])), dim=-1)

#             syns.append(syn)
#             mems.append(mem)
#             outputs.append(output)
#         return outputs, syns, mems

#     def reset(self) -> None:
#         for layer in self.bislstm:
#             layer["f"].reset()
#             layer["b"].reset()


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
        self.initialized = False

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.bislstm = nn.ModuleDict({
            "f": SpikingLayer(embedding_dim, hidden_dim, dropout),
            "b": SpikingLayer(embedding_dim, hidden_dim, dropout),
        })

        self.layers = nn.ModuleList([
            SpikingLayer(hidden_dim, hidden_dim, dropout) \
            for _ in range(num_layers)
        ])

        self.transform = nn.ModuleDict({
            "syn_fc": nn.Linear(hidden_dim * 2, hidden_dim),
            "mem_fc": nn.Linear(hidden_dim * 2, hidden_dim),
            "output_fc": nn.Linear(hidden_dim * 2, hidden_dim),
            "norm": Dyt(hidden_dim),
        })

    def init_hidden(
        self, batch_size: int, device: torch.device,
        bi_syn: torch.Tensor, bi_mem: torch.Tensor
    ) -> None:
        for _, layer in enumerate(self.layers):
            layer.init_hidden(
                batch_size, device,
                bi_syn,
                bi_mem,
            )
        self.initialized = True

    def forward(self, src: torch.Tensor) -> \
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        src = self.norm(self.embedding(src))
        f_out = src
        b_out = torch.flip(src.clone(), dims=[1])
        f_out, _, _ = self.bislstm["f"](f_out)
        b_out, _, _ = self.bislstm["b"](b_out)
        f_syn, f_mem = self.bislstm["f"].syn, self.bislstm["f"].mem
        b_syn, b_mem = self.bislstm["b"].syn, self.bislstm["b"].mem

        bi_syn = torch.cat((f_syn, b_syn), dim=-1)
        bi_syn = self.transform["syn_fc"](bi_syn)
        bi_mem = torch.cat((f_mem, b_mem), dim=-1)
        bi_mem = self.transform["mem_fc"](bi_mem)
        output = torch.cat((f_out, torch.flip(b_out, dims=[1])), dim=-1)
        output = self.transform["norm"](self.transform["output_fc"](output))

        B = src.size(0)
        device = src.device
        if self.initialized is False:
            self.init_hidden(B, device, bi_syn, bi_mem)

        syns, mems = [], []
        for _, layer in enumerate(self.layers):
            output, syn, mem = layer(output)
            # syn, mem = layer.syn, layer.mem
            syns.append(syn)
            mems.append(mem)
        return syns, mems

    def reset(self) -> None:
        self.bislstm["f"].reset()
        self.bislstm["b"].reset()
        for layer in self.layers:
            layer.reset()
        self.initialized = False


# general Luong attention
class Attention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = Dyt(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_mem, encoder_mems):
        """
        decoder_mem: (batch, hidden_dim)
        encoder_mems: (batch, seq_len, hidden_dim)
        """
        dec = self.W(decoder_mem).unsqueeze(2)                          # (batch, hidden_dim, 1)

        # Compute scores = batch matmul
        scores = torch.bmm(encoder_mems, dec).squeeze(-1)               # (batch, seq_len)

        # Softmax over encoder seq_len
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)      # (batch, seq_len)

        # Weighted sum
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

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": Attention(hidden_dim, dropout),
                "spiking": SpikingLayer(
                    embedding_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    hidden_dim, dropout
                )
            }) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, vocab_size)

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
        self, inp: torch.Tensor,
        encoder_mems: list[torch.Tensor],
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            mem = layer["spiking"].mem
            context = layer["attention"](mem, encoder_mems[i])
            inp = torch.cat((inp, context), dim=-1)
            inp = layer["spiking"](inp)
        return inp

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
                input_id = torch.argmax(self.fc(outputs[step - 1]).detach(), dim=-1)
                previous_tgt = self.norm(self.embedding(input_id.detach()))
                outputs.append(self._forward_step(previous_tgt, encoder_mems))
        outputs = torch.stack(outputs, dim=1)
        return self.fc(outputs)

    def forward(
        self, tgt: torch.Tensor,
        encoder_syns: list[torch.Tensor],
        encoder_mems: list[torch.Tensor],
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B = tgt.size(0)
        device = tgt.device
        if self.initialized is False:
            self.init_hidden(B, device, encoder_syns, encoder_mems)
        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self._forward_seq(tgt, encoder_mems, teacher_forcing_ratio)
        elif tgt.dim() == 2:
            return self.fc(self._forward_step(tgt, encoder_mems))
        else:
            raise ValueError("input dim must be 2(step) or 3(sequence)")

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

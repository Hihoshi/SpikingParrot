import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as utils
from normalizer import Dyt


class SpikingLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        residual: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual = residual

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.normalizer = Dyt(input_size=self.hidden_dim)

        self.slstm = snn.SLSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            spike_grad=snn.surrogate.atan(),
            learn_threshold=True,
            reset_mechanism="subtract",
            init_hidden=False,
        )

    def reset_hidden(self):
        return self.slstm.reset_mem()

    def forward(self, x, syn, mem):
        batch_size, seq_len, _ = x.shape
        spks = torch.zeros(
            batch_size, seq_len, self.hidden_dim,
            device=x.device, dtype=x.dtype
        )
        for step in range(seq_len):
            spk, syn, mem = self.slstm(x[:, step, :], syn, mem)
            spks[:, step] = spk

        output = self.fc(spks)
        if self.residual:
            output = self.normalizer(output + x)
        else:
            output = self.normalizer(output)
        return output, syn, mem


class SpikingParrot(nn.Module):
    def __init__(
        self,
        bidirectional: bool,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.src_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.src_normalizer = Dyt(embedding_dim)
        self.tgt_normalizer = Dyt(embedding_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            vocab_size
        )
        if bidirectional:
            self.syn_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.mem_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        enc_input_dim = embedding_dim
        dec_input_dim = embedding_dim
        for i in range(self.num_layers):
            if bidirectional:
                # 双向时需要前向和后向层
                # 第一层因为形状不匹配无法残差连接
                self.encoders.append(
                    nn.ModuleDict({
                        'forward_layer': SpikingLayer(enc_input_dim, hidden_dim, residual=False) if i == 0
                        else SpikingLayer(enc_input_dim, hidden_dim),
                        'backward_layer': SpikingLayer(enc_input_dim, hidden_dim, residual=False) if i == 0
                        else SpikingLayer(enc_input_dim, hidden_dim),
                    })
                )
                self.decoders.append(
                    SpikingLayer(dec_input_dim, hidden_dim * 2, residual=False) if i == 0
                    else SpikingLayer(dec_input_dim, hidden_dim * 2)
                )
            else:
                self.encoders.append(
                    SpikingLayer(enc_input_dim, hidden_dim, residual=False) if i == 0
                    else SpikingLayer(enc_input_dim, hidden_dim)
                )
                self.decoders.append(
                    SpikingLayer(dec_input_dim, hidden_dim, residual=False) if i == 0
                    else SpikingLayer(dec_input_dim, hidden_dim)
                )
            enc_input_dim = hidden_dim
            dec_input_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

    def reset(self):
        """reset all slstm hidden states"""
        for layer in self.encoders:
            if isinstance(layer, nn.ModuleDict):
                utils.reset(layer['forward_layer'].slstm)
                utils.reset(layer['backward_layer'].slstm)
            else:
                utils.reset(layer.slstm)
        for layer in self.decoders:
            utils.reset(layer.slstm)

    def forward(self, src, tgt):
        # encode process
        src_emb = self.src_normalizer(self.src_embedding(src))
        # if bidirectional, src needs to be flipped for backward
        if self.bidirectional:
            src_emb_flipped = torch.flip(src_emb, dims=[1])
            # encode src
            for i, layer in enumerate(self.encoders):
                # initialize hidden states
                f_syn, f_mem = layer['forward_layer'].reset_hidden()
                b_syn, b_mem = layer['backward_layer'].reset_hidden()
                # forward pass
                src_emb, f_syn, f_mem = layer['forward_layer'](src_emb, f_syn, f_mem)
                src_emb_flipped, b_syn, b_mem = layer['backward_layer'](src_emb_flipped, b_syn, b_mem)
        else:
            for layer in self.encoders:
                syn, mem = layer.reset_hidden()
                src_emb, syn, mem = layer(src_emb, syn, mem)

        # decode process
        syn = self.syn_fc(torch.cat([f_syn, b_syn], dim=1))
        mem = self.mem_fc(torch.cat([f_mem, b_mem], dim=1))
        tgt_emb = self.tgt_normalizer(self.tgt_embedding(tgt))
        for i, layer in enumerate(self.decoders):
            if i != 0:
                syn, mem = layer.reset_hidden()
            tgt_emb, syn, mem = layer(tgt_emb, syn, mem)

        return self.fc(tgt_emb)

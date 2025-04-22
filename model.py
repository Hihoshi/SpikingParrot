import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as utils
from normalizer import Dyt


class SpikingLayer(nn.Module):
    def __init__(self,
        input_dim: int, hidden_dim: int,
        residual: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.normalizer = Dyt(input_size=hidden_dim)
        self.slstm = snn.SLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            spike_grad=snn.surrogate.atan(),
            learn_threshold=True,
            reset_mechanism="subtract",
            init_hidden=False,
        )

    def reset_hidden(self):
        return self.slstm.reset_mem()

    def forward(self, x, syn=None, mem=None):
        """
        支持两种模式：
        1. 序列模式 (训练):  x.shape = (B, S, D)
        2. 单步模式 (推理):  x.shape = (B, D)
        """
        if x.dim() == 3:
            return self._forward_seq(x, syn, mem)
        elif x.dim() == 2:
            return self._forward_step(x, syn, mem)
        else:
            raise ValueError("输入维度应为2(单步)或3(序列)")

    def _forward_seq(self, x_seq, syn=None, mem=None):
        """处理完整序列 (batch, seq_len, input_dim)"""
        B, S, _ = x_seq.shape
        # preallocate space of variables
        outputs = torch.zeros(B, S, self.hidden_dim, device=x_seq.device)
        syns = torch.zeros(B, S, self.hidden_dim, device=x_seq.device)
        mems = torch.zeros(B, S, self.hidden_dim, device=x_seq.device)

        # 初始化隐藏状态
        syn = torch.zeros(B, self.hidden_dim, device=x_seq.device) \
            if syn is None else syn
        mem = torch.zeros(B, self.hidden_dim, device=x_seq.device) \
            if mem is None else mem

        for step in range(S):
            spk, syn, mem = self.slstm(x_seq[:, step, :], syn, mem)
            output = self.fc(spk)
            if self.residual:
                output = self.normalizer(output + x_seq[:, step, :])
            else:
                output = self.normalizer(output)
            outputs[:, step] = output
            syns[:, step, :] = syn
            mems[:, step, :] = mem

        return outputs, syns, mems

    def _forward_step(self, x_step, syn, mem):
        """处理单个时间步 (batch, input_dim)"""
        B, _ = x_step.shape
        syn = torch.zeros(B, self.hidden_dim, device=x_step.device) \
            if syn is None else syn
        mem = torch.zeros(B, self.hidden_dim, device=x_step.device) \
            if mem is None else mem

        spk, syn, mem = self.slstm(x_step, syn, mem)
        output = self.fc(spk)
        if self.residual:
            output = self.normalizer(output + x_step)
        else:
            output = self.normalizer(output)
        return output, syn, mem


class SpikingEncoder(nn.Module):
    def __init__(
        self, embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional:bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
                        # fc for transforming forward and backward output to proper size
                        "fc": nn.Linear(hidden_dim * 2, hidden_dim),
                        "norm": Dyt(hidden_dim),
                    })
                )
            else:
                self.layers.append(SpikingLayer(input_dim, hidden_dim, residual=(i > 0)))
            input_dim = hidden_dim

    def forward(self, src):
        src = self.norm(self.embedding(src))
        for _, layer  in enumerate(self.layers):
            if self.bidirectional:
                src_flipped = torch.flip(src, dims=[1])
                f_outputs, f_syns, f_mems = layer["f_layer"](src)
                b_outputs, b_syns, b_mems = layer["b_layer"](src_flipped)
                # flip back b_layer output to match original order
                outputs = torch.cat((f_outputs, torch.flip(b_outputs, dims=1)), dim=-1)
                outputs = layer["norm"](layer["fc"](outputs))
            else:
                outputs, syns, mems = layer(src)
            src = outputs
        if self.bidirectional:
            syn = torch.cat([f_syns[:, -1, :], b_syns[:, 0, :]], dim=-1)
            mem = torch.cat([f_mems[:, -1, :], b_mems[:, 0, :]], dim=-1)
        else:
            syn = syns[:, -1, :]
            mem = mems[:, -1, :]

        return syn, mem


class SpikingDecoder(nn.Module):
    def __init__(
        self, embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList()

        input_dim = embedding_dim
        for i in range(num_layers):
            self.layers.append(SpikingLayer(input_dim, hidden_dim, residual=(i > 0)))
            input_dim = hidden_dim

    def forward(self, tgt, syn, mem):
        """
        支持两种模式：
        1. 序列模式 (训练):  x.shape = (B, S, D)
        2. 单步模式 (推理):  x.shape = (B, D)
        """
        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self._forward_seq(tgt, syn, mem)
        elif tgt.dim() == 2:
            return self._forward_step(tgt, syn, mem)
        else:
            raise ValueError("输入维度应为2(单步)或3(序列)")
    
    def _forward_seq(self, tgt, syn, mem):
        for _, layer in enumerate(self.layers):
            outputs, syns, mems = layer(tgt, syn, mem)
            syn = syns[:, -1, :]
            mem = mems[:, -1, :]
            tgt = outputs
        
        return outputs
    
    def _forward_step(self, tgt, syn, mem):
        for _, layer in enumerate(self.layers):
            output, syn, mem = layer(tgt, syn, mem)
            tgt = output

        return output


class SpikingParrot(nn.Module):
    def __init__(
        self, embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional: bool = False,
    ):
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        super().__init__()
        self.encoder = SpikingEncoder(embedding_dim, vocab_size, hidden_dim, num_layers, bidirectional)
        if bidirectional:
            self.decoder = SpikingDecoder(embedding_dim, vocab_size, hidden_dim * 2, num_layers)
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        else:
            self.decoder = SpikingDecoder(embedding_dim, vocab_size, hidden_dim, num_layers)
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        syn, mem = self.encoder(src)
        outputs = self.decoder(tgt, syn, mem)
        return self.fc(outputs)

    def reset(self):
        for i in range(self.num_layers):
            if self.bidirectional:
                utils.reset(self.encoder.layers[i]["f_layer"].slstm)
                utils.reset(self.encoder.layers[i]["b_layer"].slstm)
            else:
                utils.reset(self.encoder.layers[i].slstm)
            utils.reset(self.decoder.layers[i].slstm)
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as utils
from normalizer import Dyt


class SpikingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.slstm = snn.SLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            spike_grad=snn.surrogate.atan(),
            learn_threshold=True,
            init_hidden=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.norm = Dyt(input_size=output_dim)
    
    def reset_hidden(self):
        syn, mem = self.slstm.reset_mem()
        return syn, mem

    def forward(self, x, syn, mem):
        spks = []
        for step in range(x.size(1)):
            spk, syn, mem = self.slstm(x[:, step, :], syn, mem)
            spks.append(spk)
        
        spks = torch.stack(spks, dim=1)
        return self.norm(self.fc(spks)), syn, mem


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
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = Dyt(self.embedding_dim)

        self.decoder = nn.ModuleList()
        if self.bidirectional:
            self.encoder_forward = nn.ModuleList()
            self.encoder_backward = nn.ModuleList()
            self.fc = nn.Linear(self.hidden_dim * 2, self.vocab_size)
            # 添加第一层
            self.encoder_forward.append(SpikingLayer(self.embedding_dim, self.hidden_dim, self.hidden_dim))
            self.encoder_backward.append(SpikingLayer(self.embedding_dim, self.hidden_dim, self.hidden_dim))
            self.decoder.append(SpikingLayer(self.embedding_dim, self.hidden_dim * 2, self.hidden_dim * 2))

        else:
            self.encoder = nn.ModuleList()
            self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
            # 添加第一层
            self.encoder.append(SpikingLayer(self.embedding_dim, self.hidden_dim, self.hidden_dim))
            self.decoder.append(SpikingLayer(self.embedding_dim, self.hidden_dim, self.hidden_dim))

        for i in range(self.num_layers - 1):
            if self.bidirectional:
                self.encoder_forward.append(SpikingLayer(self.hidden_dim, self.hidden_dim, self.hidden_dim))
                self.encoder_backward.append(SpikingLayer(self.hidden_dim, self.hidden_dim, self.hidden_dim))
                self.decoder.append(SpikingLayer(self.hidden_dim * 2, self.hidden_dim * 2, self.hidden_dim * 2))
            else:
                self.encoder.append(SpikingLayer(self.hidden_dim_dim, self.hidden_dim, self.hidden_dim))
                self.decoder.append(SpikingLayer(self.hidden_dim, self.hidden_dim, self.hidden_dim))


    def reset(self):
        for i in range(self.num_layers):
            if self.bidirectional:
                utils.reset(self.encoder_forward[i].slstm)
                utils.reset(self.encoder_backward[i].slstm)
            else:
                utils.reset(self.encoder[i].slstm)
            utils.reset(self.decoder[i].slstm)

    # 前向传播修改后：
    def forward(self, src, tgt):
        src = self.norm(self.embedding(src))
        src_flipped = torch.flip(src, dims=[1])
        # 编码源序列
        for i in range(self.num_layers):
            if self.bidirectional:
                syn_f, mem_f = self.encoder_forward[i].reset_hidden()
                syn_b, mem_b = self.encoder_forward[i].reset_hidden()
                src, syn_f, mem_f = self.encoder_forward[i](src, syn_f, mem_f)
                src_flipped, syn_b, mem_b = self.encoder_backward[i](src_flipped, syn_b, mem_b)
            else:
                syn, mem = self.encoder[i].reset_hidden()
                src, syn, mem = self.encoder[i](src, syn, mem)

        outputs = []
        # 解码目标序列
        tgt = self.norm(self.embedding(tgt))
        for i in range(self.num_layers):
            if self.bidirectional:
                syn = torch.cat((syn_f, syn_b), dim=1)
                mem = torch.cat((mem_f, mem_b), dim=1)
            if i != 0:
                syn, mem = self.decoder[i].reset_hidden()
            tgt, syn, mem = self.decoder[i](tgt, syn, mem)
        outputs.append(self.fc(tgt))
        
        return torch.stack(outputs, dim=1)

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as utils
from normalizer import Dyt


class SpikingLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, residual: bool = True):
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

    def reset_hidden(self, batch_size=1, device="cpu"):
        return self.slstm.reset_mem(batch_size=batch_size, device=device)

    def forward(self, x, syn_mem=None):
        """
        支持两种模式：
        1. 序列模式 (训练):  x.shape = (B, S, D)
        2. 单步模式 (推理): x.shape = (B, D)
        """
        if x.dim() == 3:  # 训练模式
            return self._forward_sequence(x, *syn_mem) if syn_mem else self._forward_sequence(x)
        elif x.dim() == 2:  # 推理模式
            return self._forward_step(x, *syn_mem)
        else:
            raise ValueError("输入维度应为2（单步）或3（序列）")

    def _forward_sequence(self, x_seq, syn_init=None, mem_init=None):
        """处理完整序列 (batch, seq_len, input_dim)"""
        B, S, _ = x_seq.shape
        output = torch.zeros(B, S, self.hidden_dim, device=x_seq.device)
        
        # 初始化隐藏状态
        syn = syn_init if syn_init is not None else torch.zeros(B, self.hidden_dim, device=x_seq.device)
        mem = mem_init if mem_init is not None else torch.zeros(B, self.hidden_dim, device=x_seq.device)

        for t in range(S):
            spk, syn, mem = self.slstm(x_seq[:, t], syn, mem)
            out = self.fc(spk)
            if self.residual:
                out = self.normalizer(out + x_seq[:, t])
            else:
                out = self.normalizer(out)
            output[:, t] = out
        
        return output, syn, mem

    def _forward_step(self, x_t, syn, mem):
        """处理单个时间步 (batch, input_dim)"""
        spk, syn, mem = self.slstm(x_t, syn, mem)
        out = self.fc(spk)
        if self.residual:
            out = self.normalizer(out + x_t)
        else:
            out = self.normalizer(out)
        return out, syn, mem


class SpikingParrot(nn.Module):
    def __init__(
        self, bidirectional: bool, vocab_size: int, embedding_dim: int,
        hidden_dim: int, num_layers: int
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 初始化嵌入层和归一化
        self.src_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.src_normalizer = Dyt(embedding_dim)
        self.tgt_normalizer = Dyt(embedding_dim)

        # 编码解码结构初始化
        self._init_encoder_decoder()

        # 最终分类层
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)
        
        # 双向状态融合层
        if bidirectional:
            self.syn_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.mem_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def reset(self):
        """reset all slstm hidden states"""
        for layer in self.encoders:
            if isinstance(layer, nn.ModuleDict):
                utils.reset(layer['forward_'].slstm)
                utils.reset(layer['backward_'].slstm)
            else:
                utils.reset(layer.slstm)
        for layer in self.decoders:
            utils.reset(layer.slstm)

    def _init_encoder_decoder(self):
        """动态构建编码解码结构"""
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        enc_dim = self.src_embedding.embedding_dim
        dec_dim = self.tgt_embedding.embedding_dim

        for i in range(self.num_layers):
            # 编码器层
            if self.bidirectional:
                encoder = nn.ModuleDict({
                    'forward_': SpikingLayer(enc_dim, self.hidden_dim, residual=(i > 0)),
                    'backward_': SpikingLayer(enc_dim, self.hidden_dim, residual=(i > 0))
                })
                enc_dim = self.hidden_dim  # 后续层的输入维度
            else:
                encoder = SpikingLayer(enc_dim, self.hidden_dim, residual=(i > 0))
                enc_dim = self.hidden_dim

            # 解码器层
            decoder_in_dim = dec_dim
            decoder_hid_dim = self.hidden_dim * (2 if self.bidirectional else 1)
            decoder = SpikingLayer(decoder_in_dim, decoder_hid_dim, residual=(i > 0))
            dec_dim = decoder_hid_dim  # 更新下一层输入维度

            self.encoders.append(encoder)
            self.decoders.append(decoder)

    def encode(self, src):
        """编码过程支持两种模式"""
        src_emb = self.src_normalizer(self.src_embedding(src))
        
        if self.bidirectional:
            return self._bidirectional_encode(src_emb)
        return self._unidirectional_encode(src_emb)

    def _bidirectional_encode(self, src_emb):
        """双向编码实现"""
        batch_size, seq_len, _ = src_emb.shape
        src_flipped = torch.flip(src_emb, dims=[1])

        # 存储各层最终状态
        f_syn, f_mem, b_syn, b_mem = None, None, None, None

        for layer in self.encoders:
            # 前向分支
            f_out, f_syn, f_mem = layer['forward_'](src_emb)
            # 后向分支（输入翻转）
            b_out, b_syn, b_mem = layer['backward_'](src_flipped)
            
            # 更新下一层输入
            src_emb, src_flipped = f_out, b_out

        # 合并双向状态
        syn = self.syn_fc(torch.cat([f_syn, b_syn], dim=1))
        mem = self.mem_fc(torch.cat([f_mem, b_mem], dim=1))
        return syn, mem

    def _unidirectional_encode(self, src_emb):
        """单向编码实现"""
        syn, mem = None, None
        for layer in self.encoders:
            src_emb, syn, mem = layer(src_emb)
        return syn, mem

    def decode(self, tgt, syn, mem):
        """解码过程支持两种模式"""
        tgt_emb = self.tgt_normalizer(self.tgt_embedding(tgt))
        
        for i, layer in enumerate(self.decoders):
            if i == 0:  # 第一层使用编码器状态
                out, syn, mem = layer(tgt_emb, (syn, mem))
            else:       # 其他层重置状态
                out, *_ = layer(tgt_emb)
            tgt_emb = out
        return tgt_emb

    def forward(self, src, tgt):
        # 训练时使用序列模式
        syn, mem = self.encode(src)
        output = self.decode(tgt, syn, mem)
        return self.fc(output)

    ##########################
    # 推理专用方法
    ##########################
    def init_states(self, batch_size=1, device="cpu"):
        """初始化推理状态"""
        states = {}
        # 编码器状态
        for i, layer in enumerate(self.encoders):
            if self.bidirectional:
                states[f'enc{i}_f'] = layer['forward_'].reset_hidden(batch_size, device)
                states[f'enc{i}_b'] = layer['backward_'].reset_hidden(batch_size, device)
            else:
                states[f'enc{i}'] = layer.reset_hidden(batch_size, device)
        
        # 解码器状态
        for i, layer in enumerate(self.decoders):
            states[f'dec{i}'] = layer.reset_hidden(batch_size, device)
        
        return states

    def step_decode(self, x_t, states):
        """单步解码推理"""
        # 编码阶段需要预先完成
        # 更新解码器状态
        for i, layer in enumerate(self.decoders):
            dec_key = f'dec{i}'
            x_t, states[dec_key] = layer(x_t, states[dec_key])
        return self.fc(x_t), states

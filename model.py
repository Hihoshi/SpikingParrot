import torch
import torch.nn as nn
import snntorch as snn
from normalizer import Dyt
from binarizer import sdyt


class SpikingLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        fc: bool = True, mix: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = fc
        self.mix = mix
        if fc:
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.norm = Dyt(hidden_dim)
        if mix:
            self.mixer = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.slstm = snn.SLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            spike_grad=sdyt(hidden_dim),
            learn_threshold=True,
            init_hidden=False,
        )

    def forward(self, x, syn=None, mem=None):
        """
        1. 序列处理（训练）:  x.shape = (B, S, D)
        2. 单步处理 (推理):  x.shape = (B, D)
        """
        if x.dim() == 3:
            return self._forward_seq(x, syn, mem)
        elif x.dim() == 2:
            return self._forward_step(x, syn, mem)
        else:
            ValueError("input dimension should be 2 or 3 (single step or a whole sequence)")

    def _forward_seq(self, x, syn=None, mem=None):
        """一次处理整个序列 (batch, seq_len, input_dim)"""
        B, S, _ = x.shape
        # 为张量预分配内存
        spks = torch.zeros(B, S, self.hidden_dim, device=x.device)
        # 初始化隐藏状态
        syn = torch.zeros(B, self.hidden_dim, device=x.device) \
            if syn is None else syn
        mem = torch.zeros(B, self.hidden_dim, device=x.device) \
            if mem is None else mem

        for step in range(S):
            spk, syn, mem = self.slstm(x[:, step, :], syn, mem)
            spks[:, step, :] = spk

        # 返回每一个时间步的输出和隐藏状态
        if self.fc:
            outs = self.norm(self.mixer(torch.cat([self.linear(spks), x], dim=-1))) \
                if self.mix else self.norm(self.linear(spks))
            return outs, syn, mem
        else:
            return spks, syn, mem

    def _forward_step(self, x, syn, mem):
        """一次处理一个时间步 (batch, input_dim)"""
        B, _ = x.shape
        syn = torch.zeros(B, self.hidden_dim, device=x.device) \
            if syn is None else syn
        mem = torch.zeros(B, self.hidden_dim, device=x.device) \
            if mem is None else mem

        spk, syn, mem = self.slstm(x, syn, mem)
        if self.fc:
            out = self.norm(self.mixer(torch.cat([self.linear(spk), x], dim=-1))) \
                if self.mix else self.norm(self.linear(spk))
            return out, syn, mem
        else:
            return spk, syn, mem


class SpikingEncoder(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if bidirectional:
                self.layers.append(
                    nn.ModuleDict({
                        "f_layer": SpikingLayer(
                            embedding_dim if i == 0 else hidden_dim,
                            hidden_dim,
                        ),
                        "b_layer": SpikingLayer(
                            embedding_dim if i == 0 else hidden_dim,
                            hidden_dim,
                        ),
                    })
                )
            else:
                self.layers.append(
                    SpikingLayer(
                        embedding_dim if i == 0 else hidden_dim,
                        hidden_dim,
                    )
                )

    def forward(self, src):
        src = self.norm(self.embedding(src))
        f_out = src
        b_out = torch.flip(src, dims=[1]) if self.bidirectional else None
        outputs, syns, mems = [], [], []

        for layer in self.layers:
            if self.bidirectional:
                # 前向层
                f_out, f_syn, f_mem = layer["f_layer"](f_out)
                # 后向层
                b_out, b_syn, b_mem = layer["b_layer"](b_out)
                syn = torch.cat([f_syn, b_syn], dim=-1)
                mem = torch.cat([f_mem, b_mem], dim=-1)  # [batch_size, hidden_dim * 2]
                # 重新翻转b_out对齐时间步，并拼接f_out得到最后的输出
                output = torch.cat([f_out, torch.flip(b_out, dims=[1])], dim=-1)  # [batch_size, seq_len, hidden_dim * 2]
            else:
                f_out, syn, mem = layer(f_out)
                output = f_out
            syns.append(syn)  # 返回包含每一层最后一个时间步的隐藏状态的列表
            mems.append(mem)
            outputs.append(output)
        return outputs, syns, mems


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.norm = Dyt(hidden_dim)

    def forward(self, decoder_hidden, encoder_output):
        """
        decoder_hidden:     [B, H]
        encoder_outputs:    [B, S, H]
        returns:
            context:        [B, H]
            attn_weights:   [B, S]
        """
        src_len = encoder_output.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_output), dim=2)))
        attn_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)
        return self.norm(context), attn_weights


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
        self.layers = nn.ModuleList([
            SpikingLayer(
                embedding_dim + hidden_dim if i == 0 else hidden_dim + hidden_dim,
                hidden_dim, fc=(i < num_layers - 1)
            ) for i in range(num_layers)
        ])
        self.attention = nn.ModuleList([Attention(hidden_dim) for i in range(num_layers)])
        self.hidden = None

    def _init_hidden(self, encoder_syns, encoder_mems):
        """
        Initialize hidden states from encoder's hidden states.
        Assumes one layer per encoder and decoder.
        """
        self.hidden = []
        for i in range(self.num_layers):
            # Use last step of encoder's i-th layer
            syn = encoder_syns[i]  # [B, H]
            mem = encoder_mems[i]  # [B, H]
            self.hidden.append((syn, mem))

    def forward(self, tgt, encoder_outputs, encoder_syns, encoder_mems):
        self._init_hidden(encoder_syns, encoder_mems)
        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self._forward_seq(tgt, encoder_outputs)
        elif tgt.dim() == 2:
            return self._forward_step(tgt, encoder_outputs)
        else:
            raise ValueError("Input dimension should be 2 or 3 (single step or a whole sequence)")

    def _forward_step(self, tgt, encoder_outputs):
        """
        Process one time step using attention and current hidden states.
        """
        for i in range(self.num_layers):
            # Get decoder's first layer hidden state
            syn, mem = self.hidden[i]
            context, _ = self.attention[i](mem, encoder_outputs[i])  # [B, H]
            # Combine input with context
            output = torch.cat((tgt, context), dim=-1)  # 第一层[B, E + H] 后续层[B, H + H]
            output, new_syn, new_mem = self.layers[i]._forward_step(output, syn, mem)
            tgt = output
            self.hidden[i] = (new_syn, new_mem)
        return output

    def _forward_seq(self, tgt, encoder_outputs):
        B, S, _ = tgt.shape
        outputs = torch.zeros(B, S, self.hidden_dim, device=tgt.device)
        for step in range(S):
            output = self._forward_step(tgt[:, step, :], encoder_outputs)
            outputs[:, step, :] = output

        return outputs


class SpikingParrot(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = SpikingEncoder(
            padding_idx, embedding_dim, vocab_size,
            hidden_dim, num_layers, bidirectional
        )
        self.decoder = SpikingDecoder(
            padding_idx, embedding_dim, vocab_size,
            hidden_dim * 2 if bidirectional else hidden_dim, num_layers
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, vocab_size)

    def forward(self, src, tgt):
        encoder_outputs, encoder_syns, encoder_mems = self.encoder(src)
        outputs = self.decoder(tgt, encoder_outputs, encoder_syns, encoder_mems)
        return self.fc(outputs)

    def greedy_decode(self, src, bos_token_id, eos_token_id, max_length=32):
        batch_size, _ = src.shape
        device = src.device
        _, syns, mems = self.encoder(src)
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

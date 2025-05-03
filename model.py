import torch
import torch.nn as nn
import snntorch as snn
from normalizer import Dyt
from binarizer import sdyt


class SpikingLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        dropout: float, mix: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mix = mix
        self.hidden = None

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = Dyt(hidden_dim)
        self.dropout = nn.Dropout1d(dropout)  # 只在hidden_dim上dropout
        self.slstm = snn.SLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            spike_grad=sdyt(hidden_dim),
            learn_threshold=True,
            init_hidden=False,
        )

        if mix:
            self.mixer = nn.Linear(hidden_dim + input_dim, hidden_dim)

    def init_hidden(
        self, batch_size: int, device: torch.device,
        syn_init: torch.tensor = None, mem_init: torch.tensor = None,
    ) -> None:
        # syn: synaptic, mem: membrane 分别对应传统LSTM当中的cell_state和hidden_state
        syn = torch.zeros(batch_size, self.hidden_dim, device=device) if syn_init is None else syn_init
        mem = torch.zeros(batch_size, self.hidden_dim, device=device) if mem_init is None else mem_init
        self.hidden = (syn, mem)
        return

    def _forward_step(self, x: torch.tensor) -> torch.tensor:
        syn, mem = self.hidden
        spk, syn, mem = self.slstm(x, syn, mem)
        # 更新隐藏状态
        self.hidden = (syn, mem)

        out = self.norm(self.mixer(torch.cat([self.linear(spk), x], dim=-1))) if self.mix \
            else self.norm(self.linear(spk))
        return self.dropout(out)

    def _forward_seq(self, x: torch.tensor) -> torch.tensor:
        # 预分配输出张量
        B, S, _ = x.shape
        spks = torch.zeros(B, S, self.hidden_dim, device=x.device)

        syn, mem = self.hidden
        for step in range(S):
            spk, syn, mem = self.slstm(x[:, step, :], syn, mem)
            spks[:, step, :] = spk

        self.hidden = (syn, mem)
        outputs = self.norm(self.mixer(torch.cat([self.linear(spks), x], dim=-1))) if self.mix \
            else self.norm(self.linear(spks))
        outputs = self.dropout(outputs.transpose(1, 2)).transpose(1, 2)
        return outputs

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        1. 序列处理 (训练):  x.shape = (B, S, D)
        2. 单步处理 (推理):  x.shape = (B, D)
        """
        # 初始化隐藏状态
        if self.hidden is None:
            self.init_hidden(x.size(0), x.device)
        if x.dim() == 3:
            return self._forward_seq(x)
        elif x.dim() == 2:
            return self._forward_step(x)
        else:
            ValueError("输入维度必须为2(单步)或3(序列)")

    def reset(self) -> None:
        self.hidden = None
        return


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
        b_out = torch.flip(src, dims=[1]) if self.bidirectional else None
        # 每层的输出和隐藏状态
        outputs, syns, mems = [], [], []

        for _, layer in enumerate(self.layers):
            if self.bidirectional:
                # 前向层
                f_out = layer["f_layer"](f_out)
                f_syn, f_mem = layer["f_layer"].hidden
                # 后向层
                b_out = layer["b_layer"](b_out)
                b_syn, b_mem = layer["b_layer"].hidden
                # 前后隐藏状态拼接后维度转换得到最终的隐藏状态
                syn = self.syn_fc(torch.cat([f_syn, b_syn], dim=-1))
                mem = self.mem_fc(torch.cat([f_mem, b_mem], dim=-1))
                # 重新翻转b_out对齐时间步，并拼接转换后得到最后的输出
                output = self.output_fc(torch.cat([f_out, torch.flip(b_out, dims=[1])], dim=-1))
            else:
                f_out = layer(f_out)
                syn, mem = layer.hidden
                output = f_out
            syns.append(syn)
            mems.append(mem)
            outputs.append(output)
        # 返回每一层最后一步输出和隐藏状态的列表
        return outputs, syns, mems

    def reset(self) -> None:
        for layer in self.layers:
            if self.bidirectional:
                layer["f_layer"].reset()
                layer["b_layer"].reset()
            else:
                layer.reset()
        return


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
        # 投影生成Query, Key, Value
        query = self.q_proj(decoder_hidden_state)  # [B, H]
        key = self.k_proj(encoder_outputs)           # [B, S, H]
        value = self.v_proj(encoder_outputs)       # [B, S, H]
        # 计算Query与所有Key的点积，得到注意力logits
        # [B, 1, H] x [B, H, S] -> [B, 1, S]
        energy = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))
        energy = energy.squeeze(1)  # [B, S]
        # 缩放点积（防止数值不稳定）
        d_k = query.size(-1)
        attention_weights = torch.nn.functional.softmax(energy / d_k**0.5, dim=1)
        # 根据注意力权重加权聚合Value
        context_vector = torch.bmm(attention_weights.unsqueeze(1), value).squeeze(1)  # [B, H]
        # 归一化context
        context_vector = self.norm(context_vector)
        return context_vector, attention_weights


class SpikingDecoder(nn.Module):
    def __init__(
        self, padding_idx: int,
        embedding_dim: int, vocab_size: int,
        hidden_dim: int, num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.norm = Dyt(embedding_dim)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attention": Attention(hidden_dim),
                    "spiking": SpikingLayer(
                        embedding_dim + hidden_dim if i == 0 else hidden_dim + hidden_dim,
                        hidden_dim, dropout
                    )
                })
            )

    def init_hidden(
        self, batch_size: int, device: torch.device,
        encoder_syns: torch.tensor, encoder_mems: torch.tensor
    ) -> None:
        """
        使用编码器每一层的最后一步的syn, mem初始化解码器每一层最开始syn, mem
        """
        for i, layer in enumerate(self.layers):
            layer["spiking"].init_hidden(batch_size, device, encoder_syns[i], encoder_mems[i])

    def _forward_step(self, tgt: torch.tensor) -> torch.tensor:
        """
        Process one time step using attention and current hidden states.
        """
        for i, layer in enumerate(self.layers):
            # 获取编码器隐藏层状态
            _, mem = layer["spiking"].hidden
            # 计算attention context
            context, _ = layer["attention"](mem, self.encoder_outputs[i])  # [B, H]
            # 拼接输入
            output = torch.cat((tgt, context), dim=-1)  # 第一层[B, E + H] 后续层[B, H + H]
            output = layer["spiking"](output)
            tgt = output
        return output

    def _forward_seq(self, tgt: torch.tensor) -> torch.tensor:
        B, S, _ = tgt.shape
        outputs = torch.zeros(B, S, self.hidden_dim, device=tgt.device)
        for step in range(S):
            output = self._forward_step(tgt[:, step, :])
            outputs[:, step, :] = output
        return outputs

    def forward(
        self, tgt: torch.tensor, encoder_outputs: list[torch.tensor],
        encoder_syns: list[torch.tensor], encoder_mems: list[torch.tensor]
    ) -> torch.tensor:
        B = tgt.size(0)
        device = tgt.device
        self.encoder_outputs = encoder_outputs
        if self.layers[0]["spiking"].hidden is None:
            self.init_hidden(B, device, encoder_syns, encoder_mems)

        tgt = self.norm(self.embedding(tgt))
        if tgt.dim() == 3:
            return self._forward_seq(tgt)
        elif tgt.dim() == 2:
            return self._forward_step(tgt)
        else:
            ValueError("输入维度必须为2(单步)或3(序列)")
    
    def reset(self) -> None:
        for layer in self.layers:
            layer["spiking"].reset()


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
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src: torch.tensor, tgt: torch.tensor) -> torch.tensor:
        encoder_outputs, encoder_syns, encoder_mems = self.encoder(src)
        outputs = self.decoder(tgt, encoder_outputs, encoder_syns, encoder_mems)
        return self.fc(outputs)

    def reset(self) -> None:
        self.encoder.reset()
        self.decoder.reset()

    def greedy_decode(
        self, src: torch.tensor,
        bos_token_id: int, eos_token_id: int, max_length: int
    ) -> list:
        batch_size, _ = src.shape
        device = src.device
        outputs, syns, mems = self.encoder(src)
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        for _ in range(max_length):
            current_input = input_ids[:, -1]
            output = self.decoder.forward(current_input, outputs, syns, mems)
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
        self.reset()
        return sequences

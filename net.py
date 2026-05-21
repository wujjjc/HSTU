import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.SiLU(), dropout=0.1):
        """_summary_
        mlp层，包含多个全连接层，每个全连接层后面跟一个激活函数和一个dropout层
        Args:
            input_dim (_type_): 输入维度
            hidden_dim (_type_): 隐藏层
            output_dim (_type_): 输出维度
            activation (_type_, optional): _description_. Defaults to nn.SiLU().
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        layers = []
        for dim in hidden_dim:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)

class PositionalEncoding(nn.Module):
    def compute_bucket(self, seq_len, bucket_size):
        # b4 = bucket_size // 4
        # b2 = bucket_size // 2
        # b34 = bucket_size * 3 // 4

        # i = torch.arange(seq_len).unsqueeze(1)
        # j = torch.arange(seq_len).unsqueeze(0)
        # delta = i - j  # (seq_len, seq_len)

        # abs_delta = delta.abs().float()
        # scale = torch.log(torch.tensor((seq_len - 1.0) / b4))
        # log_val = torch.floor(
        #     torch.log(abs_delta.clamp(min=b4) / b4) / scale * b4
        # ).long()

        # bucket = torch.where(
        #     (delta >= 0) & (delta < b4),       # near past
        #     delta,
        #     torch.where(
        #         (delta < 0) & (delta >= -b4),   # near future
        #         delta + b2,
        #         torch.where(
        #             delta >= b4,                 # far past
        #             (b2 + log_val).clamp(max=b34 - 1),
        #             (b34 + log_val).clamp(max=bucket_size - 1)  # far future
        #         )
        #     )
        # )
        bucket = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)  # (seq_len, seq_len)
        bucket += seq_len - 1  # shift to [0, 2*seq_len-2]
        return bucket

    def __init__(self, max_len=1024, bucket_size=128, num_heads=8):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len * 2 - 1, num_heads)
        self.time_embedding = nn.Embedding(bucket_size, num_heads)
        self.bucket_size = bucket_size
        self.num_heads = num_heads
        self.register_buffer('bucket', self.compute_bucket(max_len, bucket_size))
    
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): 时间序列的输入，形状为(batch_size, seq_len)
        Returns:
            _type_: 位置编码和时间编码的和，形状为(batch, num_heads, L, L)
        """
        pe = self.pos_embedding(self.bucket).unsqueeze(0)                                                                                                               
                                                                                                                                                                        
        # 时间差 bucketing：log2(|Δt|)，与 Meta 原版一致
        diff_sec = (x.unsqueeze(-1) - x.unsqueeze(-2)).abs().float()
        idx = (torch.log(diff_sec.clamp(min=1)) / 0.301).long().clamp(0, self.bucket_size - 1)
        te = self.time_embedding(idx)
        return (pe + te).permute(0, 3, 1, 2)  # (batch, num_heads, L, L)

class HSTU(nn.Module):
    def __init__(self, input_dim, q_dim, v_dim, num_heads, seq_len=1024, dropout=0.1):
        """
        HSTU层
        U(X), V (X), Q(X), K(X) = Split(ϕ1(f1(X))) (1)
        A(X)V(X) = ϕ2(Q(X)K(X)T + rabp,t) * V (X) (2)
        Y(X) = f2 (Norm (A(X)V (X)) ⊙ U(X)) (3)
        q_dim = k_dim u_dim = v_dim
        Args:
            input_dim (_type_): 输入维度
            q_dim (_type_): Query维度
            v_dim (_type_): Value维度
            num_heads (_type_): 注意力头数
            seq_len (int, optional): 序列长度. Defaults to 1024.
            dropout (float, optional): Dropout概率. Defaults to 0.1.
        """
        super().__init__()
        self.num_heads = num_heads
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.dropout = dropout
        # 单一权重矩阵：uvqk，与 Meta 原版一致，无 bias
        self._uvqk = nn.Parameter(
            torch.empty(
                input_dim,
                v_dim * 2 * num_heads + q_dim * num_heads * 2,
            ).normal_(mean=0, std=0.02),
        )
        self.activation = nn.SiLU()
        # 输入 LayerNorm（pre-norm）
        self._norm_input = nn.LayerNorm(input_dim)
        # 注意力输出 LayerNorm
        self._norm_attn_output = nn.LayerNorm(v_dim * num_heads)
        # f2：单一 Linear 层，无激活、无 bias 后的 dropout
        self.f2 = nn.Linear(v_dim * num_heads, input_dim)
        nn.init.xavier_uniform_(self.f2.weight)
        self.rab = PositionalEncoding(max_len=seq_len, bucket_size=128, num_heads=num_heads)
        # 预计算因果掩码（上三角 = 未来位置被屏蔽）
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool),
                diagonal=1,
            ),
        )
        
    
    def forward(self, x, mask, timestamps):
        """_summary_

        Args:
            x (_type_): batch * seq * input_dim
            mask (_type_): 序列掩码 batch * seq
            timestamps (_type_): 时间戳，batch * seq
            
        """
        B, N, D = x.shape
        # 输入 pre-norm + 线性投影 + SiLU 激活
        t = torch.mm(self._norm_input(x).view(B * N, D), self._uvqk)
        t = self.activation(t)  # SiLU
        u, q, k, v = torch.split(
            t,
            [self.v_dim * self.num_heads, self.q_dim * self.num_heads,
             self.q_dim * self.num_heads, self.v_dim * self.num_heads],
            dim=-1,
        )
        # 多头 reshape：(B, N, num_heads, head_dim)
        u = u.view(B, N, self.num_heads, self.v_dim)
        q = q.view(B, N, self.num_heads, self.q_dim)
        k = k.view(B, N, self.num_heads, self.q_dim)
        v = v.view(B, N, self.num_heads, self.v_dim)
        # Einsum 计算注意力分数：(B, num_heads, N, N)
        q = torch.permute(q, (0, 2, 1, 3))  # (B, num_heads, N, q_dim)
        k = torch.permute(k, (0, 2, 1, 3))  # (B, num_heads, N, q_dim)
        v = torch.permute(v, (0, 2, 1, 3))  # (B, num_heads, N, v_dim)
        av = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)
        # 加上相对位置 + 时间 bias
        av = av + self.rab(timestamps)
        # SiLU 激活 + 1/N 缩放
        av = F.silu(av) / N
        # 组合因果掩码和 padding 掩码（1=可关注, 0=屏蔽）
        invalid_attn_mask = (
            mask.unsqueeze(1).unsqueeze(2)
            * mask.unsqueeze(1).unsqueeze(3)
            * (1.0 - self._attn_mask[:N, :N].to(x.dtype).unsqueeze(0).unsqueeze(0))
        )
        av = av * invalid_attn_mask
        # 对 value 加权求和：(B, N, num_heads * v_dim)
        av = (av @ v).permute(0, 2, 1, 3).reshape(B, N, self.num_heads * self.v_dim)  # (B, N, num_heads * v_dim)
        # 注意力输出 Norm + U 门控
        av = self._norm_attn_output(av) * u.reshape(B, N, self.num_heads * self.v_dim)
        # dropout 放在门控输出后、f2 之前（与 Meta 原版一致）
        av = self.f2(F.dropout(av, p=self.dropout, training=self.training))
        return av


class Net(nn.Module):
    def __init__(self, item_num, embed_dim, q_dim, v_dim, num_heads, num_layers=6, seq_len=1024, dropout=0.1):
        """_summary_
        整体网络结构，包含一个 embedding 层、多个 HSTU 层，层与层之间残差连接

        Args:
            item_num: 物品数量（不含 padding_idx=0）
            embed_dim: embedding 维度
            q_dim: 每个头的 Query/Key 维度
            v_dim: 每个头的 Value/U 维度
            num_heads: 注意力头数
            num_layers: HSTU 层数
            seq_len: 最大序列长度
            dropout: Dropout 概率
        """
        super().__init__()
        self._embed_dim = embed_dim
        self.item_embedding = nn.Embedding(item_num + 1, embed_dim, padding_idx=0)
        self.emb_scale = embed_dim ** 0.5
        # 不使用绝对位置嵌入，位置信息由 HSTU 内部的相对位置 bias 提供
        self.layers = nn.ModuleList([
            HSTU(embed_dim, q_dim, v_dim, num_heads, seq_len, dropout)
            for _ in range(num_layers)
        ])
        # 只在 embedding 后加 dropout，残差分支上不额外加（dropout 已在 HSTU 内部处理）
        self.dropout = nn.Dropout(dropout)

    def get_item_embbed(self, item_id):
        """_summary_

        Args:
            item_id (_type_): 物品 id，形状为 (seq, )

        Returns:
            _type_: 物品嵌入，形状为 (seq, embed_dim)
        """
        return F.normalize(self.item_embedding(item_id), dim=-1)

    def forward(self, x, mask, timestamps):
        """_summary_

        Args:
            x: batch * seq，物品 id
            mask: batch * seq，序列掩码（1=有效, 0=padding）
            timestamps: batch * seq，时间戳

        Returns:
            batch * seq * embed_dim
        """
        x = self.item_embedding(x) * self.emb_scale  # batch * seq * embed_dim
        x = self.dropout(x)
        for layer in self.layers:
            x = x + layer(x, mask, timestamps)  # 残差连接
        return F.normalize(x, dim=-1)
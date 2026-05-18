import torch
import torch.nn as nn
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
        b4 = bucket_size // 4
        b2 = bucket_size // 2
        b34 = bucket_size * 3 // 4

        i = torch.arange(seq_len).unsqueeze(1)
        j = torch.arange(seq_len).unsqueeze(0)
        delta = i - j  # (seq_len, seq_len)

        abs_delta = delta.abs().float()
        scale = torch.log(torch.tensor((seq_len - 1.0) / b4))
        log_val = torch.floor(
            torch.log(abs_delta.clamp(min=b4) / b4) / scale * b4
        ).long()

        bucket = torch.where(
            (delta >= 0) & (delta < b4),       # near past
            delta,
            torch.where(
                (delta < 0) & (delta >= -b4),   # near future
                delta + b2,
                torch.where(
                    delta >= b4,                 # far past
                    (b2 + log_val).clamp(max=b34 - 1),
                    (b34 + log_val).clamp(max=bucket_size - 1)  # far future
                )
            )
        )
        return bucket

    def __init__(self, max_len=1024, bucket_size=32, num_heads=8):
        super().__init__()
        self.pos_embedding = nn.Embedding(bucket_size, num_heads)
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
        pe = self.pos_embedding(self.bucket).unsqueeze(0)  # (1, L, L, num_heads)

        diff = x.unsqueeze(-1) - x.unsqueeze(-2)           # (batch, L, L), 有符号时间差
        abs_log = torch.log(diff.abs() + 1.0)               # log(|Δt| + 1), 0 映射到 0
        min_val = abs_log.amin(dim=(1, 2), keepdim=True)
        max_val = abs_log.amax(dim=(1, 2), keepdim=True)
        norm = (abs_log - min_val) / (max_val - min_val + 1e-8) * (self.bucket_size - 1)  # 归一化到 [0, 31]

        idx = norm.long().clamp(0, self.bucket_size - 1)
        te = self.time_embedding(idx)                       # (batch, L, L, num_heads)
        return (pe + te).permute(0, 3, 1, 2)                # (batch, num_heads, L, L)

class HSTU(nn.Module):
    def __init__(self, input_dim, q_dim, v_dim, num_heads, seq_len=1024, dropout=0.1):
        """
        HSTU层
        U(X), V (X), Q(X), K(X) = Split(ϕ1(f1(X))) (1)
        A(X)V (X) = ϕ2(Q(X)K(X)T + rabp,t) * V (X) (2)
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
        self.dropout = nn.Dropout(dropout)
        layer1 = []
        layer1.append(nn.Linear(input_dim, 2 * (q_dim + v_dim), bias=True))
        layer1.append(nn.SiLU())
        layer1.append(nn.Dropout(dropout))
        self.f1 = nn.Sequential(*layer1)
        layer2 = []
        layer2.append(nn.Linear(v_dim, input_dim, bias=True))
        layer2.append(nn.SiLU())
        layer2.append(nn.Dropout(dropout))
        self.f2 = nn.Sequential(*layer2)
        self.norm = nn.LayerNorm(v_dim)
        self.rab = PositionalEncoding(max_len=seq_len, bucket_size=32, num_heads=num_heads)
        
    
    def forward(self, x, mask, timestamps):
        """_summary_

        Args:
            x (_type_): batch * seq * input_dim
            mask (_type_): 序列掩码 batch * seq
            timestamps (_type_): 时间戳，batch * seq
            
        """
        t = self.f1(x)  # batch * seq * 2(q_dim + v_dim)
        u, q, k, v = torch.split(t, [self.v_dim, self.q_dim, self.q_dim, self.v_dim], dim=-1)  # batch * seq * v_dim/q_dim
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.q_dim // self.num_heads).transpose(1, 2)  # batch * num_heads * seq * q_dim/num_heads
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.q_dim // self.num_heads).transpose(1, 2)  # batch * num_heads * seq * q_dim/num_heads
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.v_dim // self.num_heads).transpose(1, 2)  # batch * num_heads * seq * v_dim/num_heads
        av = q @ k.transpose(-2, -1) + self.rab(timestamps)  # batch * num_heads * seq * seq
        av = av.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        av = av.masked_fill(mask.unsqueeze(1).unsqueeze(3) == 0, float(0))
        causal_mask = torch.tril(torch.ones((av.shape[-2], av.shape[-1]), device=av.device)).unsqueeze(0).unsqueeze(0) # (1, 1, seq, seq)
        av = av.masked_fill(causal_mask == 0, float('-inf')) # (batch, num_heads, seq, seq), 因果掩码填充
        av = av @ v  # batch * num_heads * seq * v_dim/num_heads
        av = av.view(av.shape[0], av.shape[2], self.v_dim)  # batch * seq * v_dim
        av = self.norm(av)  # batch * seq * v_dim
        av = av * u  # batch * seq * v_dim
        av = av * mask.unsqueeze(-1) # batch * seq * v_dim
        av = self.f2(av)  # batch * seq * input_dim
        return av


class Net(nn.Module):
    def __init__(self, item_num, embbed_dim, q_dim, v_dim, num_heads, num_layers, seq_len=1024, dropout=0.1):
        """_summary_
        整体网络结构，包含一个embbeding层,多个HSTU层，每个HSTU层的输入是前一层的输出，并且每层的输出都与输入进行残差

        Args:
            item_num (_type_): 物品数量
            embbed_dim (_type_): embedding维度
            q_dim (_type_): Query维度
            v_dim (_type_): Value维度
            num_heads (_type_): 注意力头数
            num_layers (_type_): HSTU层数
            seq_len (int, optional): 序列长度. Defaults to 1024.
            dropout (float, optional): Dropout概率. Defaults to 0.1.
        """
        super().__init__()
        self.item_embedding = nn.Embedding(item_num + 1, embbed_dim)
        self.layers = nn.ModuleList([HSTU(embbed_dim, q_dim, v_dim, num_heads, seq_len, dropout) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embbed_dim) for _ in range(num_layers)])
        
    
    def resdual_connection(self, x, net, norm):
        return x + net(norm(x))
    
    def forward(self, x, mask, timestamps):
        x = self.item_embedding(x)  # batch * seq * embbed_dim
        for i in range(len(self.layers)):
            x = self.resdual_connection(x, self.layers[i], self.norm_layers[i])
        return x  # batch * seq * embbed_dim
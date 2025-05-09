import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Inject positional information into input sequences using sine and cosine functions.
    
    Args:
        d_model: Dimension of the model embeddings
        dropout: Dropout probability
        max_len: Maximum length of input sequences (default: 4096)
    """
    def __init__(self, d_model, dropout, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 缓冲区 不更新参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert self.pe.size(1) >= x.size(1), 'max_len < seq_len!'
        
        # (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LinearSelfAttention(nn.Module):
    """
    修改后的线性自注意力层，支持任意d_input维度
    """
    def __init__(self, d_model, n_head, dropout=0.1,
                 k_dim=None, share_kv=False, share_projection=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        # 投影维度(k)，默认为d_k
        self.k_dim = k_dim if k_dim is not None else self.d_k
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # K和V的投影矩阵E和F
        if share_projection:
            # 层间共享 - 所有层/头共享单个投影矩阵
            self.E_proj = nn.Parameter(torch.Tensor(self.k_dim, self.d_model))
            self.F_proj = nn.Parameter(torch.Tensor(self.k_dim, self.d_model)) if not share_kv else self.E_proj
        else:
            # 头间共享 - 每个头单独投影矩阵
            self.E_proj = nn.Parameter(torch.Tensor(n_head, self.k_dim, self.d_k))
            if share_kv:
                self.F_proj = self.E_proj  # K-V共享
            else:
                self.F_proj = nn.Parameter(torch.Tensor(n_head, self.k_dim, self.d_k))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        
        # 初始化E和F为正态分布
        if hasattr(self.E_proj, 'dim'):
            if self.E_proj.dim() == 2:  # 层间共享
                nn.init.normal_(self.E_proj, mean=0, std=1/math.sqrt(self.k_dim))
                if not torch.equal(self.E_proj, self.F_proj):
                    nn.init.normal_(self.F_proj, mean=0, std=1/math.sqrt(self.k_dim))
            else:  # 头间共享
                nn.init.normal_(self.E_proj, mean=0, std=1/math.sqrt(self.k_dim))
                if not torch.equal(self.E_proj, self.F_proj):
                    nn.init.normal_(self.F_proj, mean=0, std=1/math.sqrt(self.k_dim))
        
        # 初始化偏置
        nn.init.constant_(self.W_Q.bias, 0.)
        nn.init.constant_(self.W_K.bias, 0.)
        nn.init.constant_(self.W_V.bias, 0.)
        nn.init.constant_(self.W_O.bias, 0.)
    
    def forward(self, x, padding_mask=None):
        """
        输入:
            x: (batch_size, seq_len, d_model)
            padding_mask: (batch_size, seq_len), 1表示padding位置
        输出:
            (batch_size, seq_len, d_model)
        """
        # (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        # --> (batch_size, seq_len, n_head, d_k) --> (batch_size, n_head, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        
        # --> (batch_size, n_head, d_k, seq_len)
        # torch.Tensor(n_head, self.k_dim, self.d_k)
        # --> (batch_size, n_head, k_dim, d_k)
        K.transpose(2, 3)
        
        # (batch_size, n_head, seq_len, seq_len)
        scores = torch.matmul(Q / math.sqrt(self.d_k), K.transpose(-2, -1))
        attn_weight = torch.softmax(scores)

        # Apply dropout
        attn_weight = self.dropout(attn_weight)

        # (batch_size, n_head, seq_len, d_v)
        attn = torch.matmul(attn_weight, V)

        # --> (batch_size, seq_len, n_head, d_v) --> (batch_size, seq_len, d_model)
        output = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch_size, seq_len, d_model)
        output = self.W_O(output)
        return output

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    """
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class LinformerLayer(nn.Module):
    """
    Linformer层
    """
    def __init__(self, d_model, n_head, d_ffn, dropout, 
                 k_dim=None, share_kv=False, share_projection=False):
        super().__init__()
        self.self_attn = LinearSelfAttention(
            d_model, n_head, dropout, k_dim, share_kv, share_projection
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask=None):
        # 自注意力 + 残差
        attn_output = self.self_attn(x, padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # FFN + 残差
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class Linformer(nn.Module):
    """
    修改后的Linformer，支持(batch_size, seq_len, d_input)输入
    """
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers,
                 d_input, d_output, dropout, k_dim=None, 
                 share_kv=False, share_projection=False):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([LinformerLayer(d_model, n_head, d_ffn, dropout, k_dim, share_kv, share_projection)
                                        for _ in range(num_encoder_layers)])
        self.out = nn.Linear(d_model, d_output)

        
        # 初始化权重
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
    def forward(self, x, padding_mask=None):
        """
        输入:
            x: (batch_size, seq_len, d_input)
            padding_mask: (batch_size, seq_len), 1表示padding位置
        输出:
            (batch_size, seq_len, d_input)
        """
        # (batch_size, seq_len, d_input)
        x = self.embedding(x)
        # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        # (batch_size, d_model)
        output = x.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output

# 使用示例
if __name__ == "__main__":

    x = torch.arange(40).reshape(2, 4, 5)
    z = torch.arange(20).reshape(5, 4)
    # y = torch.arange(720).reshape(2, 3, 5, 4)
    x = torch.matmul(x, z)
    print(x.shape)
    # 参数配置
    d_input = 7
    d_output = 7
    d_model = 512
    num_encoder_layers = 6
    n_head = 8
    d_ffn = 3072  # 通常为4*d_input
    dropout = 0.1
    k_dim = 128  # 投影维度
    share_kv = True  # 共享K和V投影
    share_projection = True  # 层间共享投影矩阵
    
    # 创建模型
    model = Linformer(
        d_model, n_head, d_ffn, num_encoder_layers,
        d_input, d_input, dropout, k_dim, share_kv, share_projection
    )
    
    # 示例输入 (batch_size=32, seq_len=128, d_input=768)
    batch_size = 32
    seq_len = 128
    example_input = torch.randn(batch_size, seq_len, d_input)

    # 前向传播
    output = model(example_input)
    print("输出形状:", output.shape)  # 应该是 (32, 7)
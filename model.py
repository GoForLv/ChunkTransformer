import torch
from torch import nn

import math

def softmax(scores: torch.Tensor, dim=-1):
    """Compute softmax with improved numerical stability.
    
    Args:
        scores: Input tensor containing unnormalized scores
        dim: Dimension along which softmax will be computed (default: -1)
    
    Returns:
        Tensor with same shape as input, with softmax applied along specified dimension
    """
    scores_max, _ = torch.max(scores, dim=dim, keepdim=True)
    scores_exp = torch.exp(scores - scores_max)
    attn = scores_exp / (scores_exp.sum(dim=dim, keepdim=True) + 1e-10)
    return attn

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
        # (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class HierarchicalBlockAttention(nn.Module):
    """Hierarchical block attention mechanism that processes input in blocks.
    
    Args:
        d_model: Dimension of input embeddings
        nhead: Number of attention heads
        dropout: Dropout probability
        d_block: Size of each processing block
    """
    def __init__(self, d_model, nhead, dropout, d_block):
        super(HierarchicalBlockAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_v = self.d_k
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Initialize weights, gain adapt to ReLU
        nn.init.xavier_uniform_(self.W_Q.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W_K.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W_V.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W_O.weight, gain=math.sqrt(2))

        self.dropout = nn.Dropout(dropout)
        self.d_block = d_block

    def forward(self, q, k, v):
        # (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = q.size()
        # (batch_size, seq_len, d_model)
        Q, K, V = self.W_Q(q), self.W_K(k), self.W_V(v)

        # (batch_size, nhead, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_v).transpose(1, 2)

        # 分块后: (batch_size, nhead, nblock, d_block, d_k)
        Q = Q.view(batch_size, self.nhead, -1, self.d_block, self.d_k)
        K = K.view(batch_size, self.nhead, -1, self.d_block, self.d_k)
        V = V.view(batch_size, self.nhead, -1, self.d_block, self.d_v)
        # (batch_size, nhead, nblock, d_block, d_block)
        scores = torch.matmul(Q / math.sqrt(self.d_k), K.transpose(-2, -1))
        attn_weight = softmax(scores)

        # Apply dropout
        attn_weight = self.dropout(attn_weight)

        # (batch_size, nhead, nblock, d_block, d_v)
        attn = torch.matmul(attn_weight, V)
        # (batch_size, nhead, seq_len, d_v)
        concat = attn.view(batch_size, self.nhead, -1, self.d_v)

        # (batch_size, seq_len, d_model)
        concat = concat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch_size, seq_len, d_model)
        output = self.W_O(concat)
        return output

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention mechanism.
    
    Args:
        d_model: Dimension of input embeddings
        nhead: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, d_model, nhead, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_v = self.d_k
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Initialize weights, gain adapt to ReLU
        nn.init.xavier_uniform_(self.W_Q.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W_K.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W_V.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W_O.weight, gain=math.sqrt(2))
    
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        # (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        # (batch_size, seq_len, d_model)
        Q, K, V = self.W_Q(q), self.W_K(k), self.W_V(v)

        # (batch_size, nhead, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_v).transpose(1, 2)
        # (batch_size, nhead, seq_len, seq_len)
        scores = torch.matmul(Q / math.sqrt(self.d_k), K.transpose(-2, -1))
        attn_weight = softmax(scores)

        # Apply dropout
        attn_weight = self.dropout(attn_weight)

        # (batch_size, nhead, seq_len, d_v)
        attn = torch.matmul(attn_weight, V)

        # (batch_size, seq_len, d_model)
        concat = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch_size, seq_len, d_model)
        output = self.W_O(concat)
        return output

class FeedForward(nn.Module):
    """Position-wise feed forward network with ReLU activation.
    
    Args:
        d_model: Dimension of input and output
        d_ffn: Dimension of hidden layer
        dropout: Dropout probability
    """
    def __init__(self, d_model, d_ffn, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
    
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.fc2.weight, gain=math.sqrt(2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        # 激活函数后
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Single layer of Transformer encoder with either standard or hierarchical attention."""
    def __init__(self, d_model, nhead, d_ffn, dropout, d_block, attn):
        super(TransformerEncoderLayer, self).__init__()
        if attn == 'Base':
            self.attn = MultiHeadAttention(d_model, nhead, dropout)
        elif attn == 'HBA':
            self.attn = HierarchicalBlockAttention(d_model, nhead, dropout, d_block)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        # two SubLayerNorm：一个用于注意力后，一个用于FFN后
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN: SubLayer(x) = x + Dropout(Sublayer(LayerNorm(x)))
        x_norm1 = self.norm1(x)
        attn = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + self.dropout(attn)

        x_norm2 = self.norm2(x)
        ffn = self.ffn(x_norm2)
        x = x + self.dropout(ffn)
        return x

class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""
    def __init__(self, d_model, nhead, d_ffn, num_encoder_layers, dropout, d_block, attn):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList()
        for i in range(num_encoder_layers):
            self.encoders.add_module('encoder'+str(i),
                                     TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, d_block, attn))

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class Transformer(nn.Module):
    """Complete Transformer model with custom implementation."""
    def __init__(self, d_model, nhead, d_ffn, num_encoder_layers, d_input, d_output, dropout, d_block, attn):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, d_ffn, num_encoder_layers, dropout, d_block, attn)
        self.out = nn.Linear(d_model, d_output)

        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0, std=0.02)

        self.d_block = d_block
        self.attn = attn

    def forward(self, x):
        # (batch_size, seq_len, d_input)
        origin_seq_len = x.size(1)
        if origin_seq_len % self.d_block != 0 and self.attn == 'HBA':
            pad_len = self.d_block - (origin_seq_len % self.d_block)
            x = nn.functional.pad(x, (0, pad_len, 0), value=0)

        # (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # (batch_size, d_model)
        output = output[:, -1, :]
        # (batch_size, d_output)
        output = self.out(output)
        return output

class TorchTransformer(nn.Module):
    """Transformer model using PyTorch's built-in components."""
    def __init__(self, d_model, nhead, d_ffn, num_encoder_layers, d_input, d_output, dropout):
        super(TorchTransformer, self).__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.out = nn.Linear(d_model, d_output)
    
        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0, std=0.02)

    def forward(self, x):
        # src形状: (batch_size, seq_len, d_input)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # (batch_size, d_model)
        output = output[:, -1, :]
        # (batch_size, d_output)
        output = self.out(output)
        return output
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
    attn = scores_exp / (scores_exp.sum(dim=dim, keepdim=True) + 1e-8)
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
        assert self.pe.size(1) >= x.size(1), 'max_len < seq_len!'
        
        # (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    """
    Full Attention.
    """
    def __init__(self, d_model, n_head, dropout):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Initialize weights, gain adapt to ReLU
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        Q, K, V = self.W_Q(q), self.W_K(k), self.W_V(v)

        # --> (batch_size, seq_len, n_head, d_k) --> (batch_size, n_head, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # (batch_size, n_head, seq_len, seq_len)
        scores = torch.matmul(Q / math.sqrt(self.d_k), K.transpose(-2, -1))
        attn_weight = softmax(scores)

        # Apply dropout
        attn_weight = self.dropout(attn_weight)

        # (batch_size, n_head, seq_len, d_v)
        attn = torch.matmul(attn_weight, V)

        # --> (batch_size, seq_len, n_head, d_v) --> (batch_size, seq_len, d_model)
        output = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch_size, seq_len, d_model)
        output = self.W_O(output)
        return output

class BlockAttention(nn.Module):
    """Hierarchical block attention mechanism that processes input in blocks.
    
    Args:
        d_model: Dimension of input embeddings
        n_head: Number of attention heads
        dropout: Dropout probability
        d_block: Size of each processing block
    """
    def __init__(self, d_model, n_head, dropout, d_block):
        super(BlockAttention, self).__init__()
        self.d_model = d_model
        self.d_block = d_block
        self.full_attn = MultiheadAttention(d_model, n_head, dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model)
        batch_size = x.size(0)

        # 分块
        # --> (batch_size, n_block, d_block, d_model) --> (batch_size * n_block, d_block, d_model)
        x = x.view(-1, self.d_block, self.d_model)

        x = self.full_attn(x, x, x)
        
        # --> (batch_size, n_block, d_block, d_model) --> (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.d_model)
        return x

class SelfAttention(nn.Module):
    """Standard multi-head attention mechanism.
    
    Args:
        d_model: Dimension of input embeddings
        n_head: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, d_model, n_head, dropout):
        super(SelfAttention, self).__init__()
        self.full_attn = MultiheadAttention(d_model, n_head, dropout)
    
    def forward(self, x):
        x = self.full_attn(x, x, x)
        return x

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
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

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
    def __init__(self, d_model, n_head, d_ffn, dropout, d_block, attn):
        super(TransformerEncoderLayer, self).__init__()
        if attn == 'Base':
            self.attn = SelfAttention(d_model, n_head, dropout)
        elif attn == 'HBA':
            self.attn = BlockAttention(d_model, n_head, dropout, d_block)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        # two SubLayerNorm：一个用于注意力后，一个用于FFN后
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN: SubLayer(x) = x + Dropout(Sublayer(LayerNorm(x)))
        x_norm1 = self.norm1(x)
        attn = self.attn(x_norm1)
        x = x + self.dropout(attn)

        x_norm2 = self.norm2(x)
        ffn = self.ffn(x_norm2)
        x = x + self.dropout(ffn)
        return x

class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, dropout, d_block, attn):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList()
        for i in range(num_encoder_layers):
            self.encoders.add_module('encoder'+str(i),
                                     TransformerEncoderLayer(d_model, n_head, d_ffn, dropout, d_block, attn))

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class Transformer(nn.Module):
    """Complete Transformer model with custom implementation."""
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, d_input, d_output, dropout, d_block, attn):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = TransformerEncoder(d_model, n_head, d_ffn, num_encoder_layers, dropout, d_block, attn)
        self.out = nn.Linear(d_model, d_output)

        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out.weight)

        self.d_block = d_block
        self.attn = attn

    def forward(self, x):
        # (batch_size, seq_len, d_input)
        x = self.embedding(x)
        # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # (batch_size, d_model)
        output = output.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output

class TorchTransformer(nn.Module):
    """Transformer model using PyTorch's built-in components."""
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, d_input, d_output, dropout):
        super(TorchTransformer, self).__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ffn, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.out = nn.Linear(d_model, d_output)
    
        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        # (batch_size, seq_len, d_input)
        # (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # (batch_size, d_model)
        output = output.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output
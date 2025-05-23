import torch
from torch import nn

import math

from utils import softmax, PatchEmbedding, PositionalEncoding

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

        self.init_parameters()
        self.dropout = nn.Dropout(dropout)

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

        nn.init.constant_(self.W_Q.bias, 0.)
        nn.init.constant_(self.W_K.bias, 0.)
        nn.init.constant_(self.W_V.bias, 0.)
        nn.init.constant_(self.W_O.bias, 0.)

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

        self.init_parameters()
        self.dropout = nn.Dropout(dropout)

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        # 激活函数后
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PoolingLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout) -> None:
        super(PoolingLayer, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = MultiheadAttention(d_model, n_head, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # (batch_size * n_block, d_block, d_model)
        batch_size = x.size(0)

        # (batch_size * n_block, 1, d_model)
        query = self.query.expand(batch_size, -1, -1)
        
        x_norm = self.norm(x)
        # (batch_size * n_block, 1, d_model)
        x = self.attn(query, x_norm, x_norm)
        x = self.dropout(x)
        # (batch_size * n_block, d_model)
        return x.squeeze(1)

class TransformerEncoderLayer(nn.Module):
    """Single layer of Transformer encoder with either standard or hierarchical attention."""
    def __init__(self, d_model, n_head, d_ffn, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiheadAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        # two SubLayerNorm：一个用于注意力后，一个用于FFN后
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN: SubLayer(x) = x + Dropout(Sublayer(LayerNorm(x)))
        x_norm1 = self.norm1(x)
        attn = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + self.dropout1(attn)

        x_norm2 = self.norm2(x)
        ffn = self.ffn(x_norm2)
        x = x + self.dropout2(ffn)
        return x

class HBATransformer(nn.Module):
    """Complete Transformer model with custom implementation."""
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, d_input, d_output, d_block, dropout):
        super(HBATransformer, self).__init__()
        self.d_input = d_input
        self.patch_embedding = PatchEmbedding(embed_dim=d_model, dropout=dropout)
        if d_input != 0:
            self.embedding = nn.Linear(d_input, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.local_layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, d_ffn, dropout)
                                           for _ in range(num_encoder_layers)])
        self.pooling = PoolingLayer(d_model, n_head, dropout)
        self.global_layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, d_ffn, dropout)
                                           for _ in range(num_encoder_layers)])
        self.out = nn.Linear(d_model, d_output)

        self.init_parameters()

        self.d_block = d_block
        self.d_model = d_model

    def init_parameters(self):
        if self.d_input != 0:
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.constant_(self.embedding.bias, 0.)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x):
        # (batch_size, seq_len, d_input)
        batch_size = x.size(0)

        # ViT: (batch_size, channels, H, W)
        if len(x.shape) == 4:
            x = self.patch_embedding(x)
        # LSTF: (batch_size, seq_len, d_input)
        else:
            x = self.embedding(x)

        # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        # (batch_size * n_block, d_block, d_model)
        x = x.contiguous().view(-1, self.d_block, self.d_model)
        for layer in self.local_layers:
            x = layer(x)

        # (batch_size * n_block, d_model)
        x = self.pooling(x)

        # (batch_size, n_block, d_model)
        x = x.view(batch_size, -1, self.d_model)
        for layer in self.global_layers:
            x = layer(x)

        # (batch_size, d_model)
        output = x.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output

class LocalHBATransformer(nn.Module):
    """Complete Transformer model with custom implementation."""
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, d_input, d_output, d_block, dropout):
        super(LocalHBATransformer, self).__init__()
        self.d_input = d_input
        self.patch_embedding = PatchEmbedding(embed_dim=d_model, dropout=dropout)
        if d_input != 0:
            self.embedding = nn.Linear(d_input, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.local_layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, d_ffn, dropout)
                                           for _ in range(num_encoder_layers)])
        self.out = nn.Linear(d_model, d_output)

        self.init_parameters()

        self.d_block = d_block
        self.d_model = d_model

    def init_parameters(self):
        if self.d_input != 0:
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.constant_(self.embedding.bias, 0.)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x):
        # (batch_size, seq_len, d_input)
        batch_size = x.size(0)
        # ViT: (batch_size, channels, H, W)
        if len(x.shape) == 4:
            x = self.patch_embedding(x)
        # LSTF: (batch_size, seq_len, d_input)
        else:
            x = self.embedding(x)

        # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        # (batch_size * n_block, d_block, d_model)
        x = x.contiguous().view(-1, self.d_block, self.d_model)
        for layer in self.local_layers:
            x = layer(x)

        # (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.d_model)
        # (batch_size, d_model)
        output = x.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output

class BaseTransformer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, d_input, d_output, dropout):
        super(BaseTransformer, self).__init__()
        self.d_input = d_input
        self.patch_embedding = PatchEmbedding(embed_dim=d_model, dropout=dropout)
        if d_input != 0:
            self.embedding = nn.Linear(d_input, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.encoders = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, d_ffn, dropout)
                                           for _ in range(num_encoder_layers)])
        self.out = nn.Linear(d_model, d_output)

        self.init_parameters()

    def init_parameters(self):
        if self.d_input != 0:
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.constant_(self.embedding.bias, 0.)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x):
        # ViT: (batch_size, channels, H, W)
        if len(x.shape) == 4:
            x = self.patch_embedding(x)
        # LSTF: (batch_size, seq_len, d_input)
        else:
            x = self.embedding(x)

        # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        for encoder in self.encoders:
            x = encoder(x)
        # (batch_size, d_model)
        output = x.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output

class TorchTransformer(nn.Module):
    """Transformer model using PyTorch's built-in components."""
    def __init__(self, d_model, n_head, d_ffn, num_encoder_layers, d_input, d_output, dropout):
        super(TorchTransformer, self).__init__()
        self.d_input = d_input
        self.patch_embedding = PatchEmbedding(embed_dim=d_model, dropout=dropout)
        if d_input != 0:
            self.embedding = nn.Linear(d_input, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ffn, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.out = nn.Linear(d_model, d_output)
    
        self.init_parameters()
    
    def init_parameters(self):
        if self.d_input != 0:
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.constant_(self.embedding.bias, 0.)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x):
        # ViT: (batch_size, channels, H, W)
        if len(x.shape) == 4:
            x = self.patch_embedding(x)
        # LSTF: (batch_size, seq_len, d_input)
        else:
            x = self.embedding(x)

        # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        # (batch_size, d_model)
        output = x.mean(dim=1)
        # (batch_size, d_output)
        output = self.out(output)
        return output
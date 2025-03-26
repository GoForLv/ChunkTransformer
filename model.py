import torch
from torch import nn

from utils import Counter

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
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
        # (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 缓冲区 不更新参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, n_neighbor, d_chunk, max_len=1024):
        super(MyAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 缓冲区 局部注意力掩码
        # self.n_neighbor = n_neighbor
        # self.register_buffer("local_mask", self.create_local_mask(max_len))
        # 缓冲区 分块注意力掩码
        self.d_chunk = d_chunk
        # self.register_buffer("chunk_mask", self.create_chunk_mask(max_len))
    
    # def create_local_mask(self, max_len):
    #     """局部注意力掩码"""
    #     mask = torch.zeros(max_len, max_len, dtype=torch.bool)
    #     for i in range(max_len):
    #         start = max(0, i - self.n_neighbor)
    #         end = min(max_len, i + self.n_neighbor + 1)
    #         mask[i, start:end] = True
    #     return mask  # [max_len, max_len]

    # def create_chunk_mask(self, max_len):
    #     """分块注意力掩码"""
    #     mask = torch.zeros(max_len, max_len, dtype=torch.bool)
    #     for i in range(0, max_len, self.d_chunk):
    #         mask[i:i+self.d_chunk, i:i+self.d_chunk] = True
    #     return mask  # [max_len, max_len]

    def forward(self, q, k, v):
        # (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = q.size()
        # (batch_size, seq_len, d_model)
        Q, K, V = self.W_Q(q), self.W_K(k), self.W_V(v)
        # (batch_size, nhead, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # 分块后: (batch_size, nhead, nchunk, d_chunk, d_k)
        Q = Q.view(batch_size, self.nhead, -1, self.d_chunk, self.d_k)
        K = K.view(batch_size, self.nhead, -1, self.d_chunk, self.d_k)
        V = V.view(batch_size, self.nhead, -1, self.d_chunk, self.d_k)
        # (batch_size, nhead, nchunk, d_chunk, d_chunk)
        scaled_dot = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 局部注意力稀疏掩码
        # scaled_dot.masked_fill_(~self.local_mask[:seq_len, :seq_len], float('-inf'))
        # 分块注意力稀疏掩码
        # scaled_dot.masked_fill_(~self.chunk_mask[:seq_len, :seq_len], float('-inf'))

        attn_weight = torch.softmax(scaled_dot, dim=-1)
        # 注意力权重
        attn_weight = self.dropout(attn_weight)
        attn = torch.matmul(attn_weight, V)
        # (batch_size, nhead, seq_len, d_k)
        concat = attn.view(batch_size, self.nhead, -1, self.d_k)

        # (batch_size, seq_len, d_model)
        concat = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch_size, seq_len, d_model)
        output = self.W_O(concat)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        # (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        # (batch_size, seq_len, d_model)
        Q, K, V = self.W_Q(q), self.W_K(k), self.W_V(v)
        # (batch_size, nhead, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        # (batch_size, nhead, seq_len, seq_len)
        scaled_dot = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weight = torch.softmax(scaled_dot, dim=-1)
        # 注意力权重
        attn_weight = self.dropout(attn_weight)
        # (batch_size, nhead, seq_len, d_k)
        attn = torch.matmul(attn_weight, V)
        # (batch_size, seq_len, d_model)
        concat = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch_size, seq_len, d_model)
        output = self.W_O(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        # 激活函数后
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        # 残差连接
        add_norm = self.norm(x + self.dropout(y))
        return add_norm

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dropout, n_neighbor, d_chunk, is_original):
        super(Encoder, self).__init__()
        if is_original:
            self.attn = MultiHeadAttention(d_model, nhead, dropout)
        else:
            self.attn = MyAttention(d_model, nhead, dropout, n_neighbor, d_chunk)

        self.ffn = FeedForward(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        attn = self.attn(x, x, x)
        x = self.add_norm1(x, attn)

        ffn = self.ffn(x)
        x = self.add_norm2(x, ffn)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, num_encoder_layers, dropout, n_neighbor, d_chunk, is_original):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.Sequential()
        for i in range(num_encoder_layers):
            self.encoders.add_module('encoder'+str(i), Encoder(d_model, nhead, d_ffn, dropout, n_neighbor, d_chunk, is_original))

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, num_encoder_layers, d_input, d_output, dropout, n_neighbor, d_chunk, is_original):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, d_ffn, num_encoder_layers, dropout, n_neighbor, d_chunk, is_original)
        self.fc = nn.Linear(d_model, d_output)

    def forward(self, x):
        # (batch_size, seq_len, d_input)
        x
        # (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(x)
        # (batch_size, d_model)
        output = output[:, -1, :]
        # (batch_size, d_output)w
        output = self.fc(output)
        return output

class TorchTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, num_encoder_layers, d_input, d_output, dropout):
        super(TorchTransformer, self).__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ffn, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, d_output)

    def forward(self, src):
        # src形状: (batch_size, seq_len, d_input)
        src = self.embedding(src)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)
        output = output[-1]  # 取最后一个时间步的输出 (batch_size, d_model)
        output = self.fc(output)  # (batch_size, pred_len)
        return output

class RNN(nn.Module):
    '''
    A simple RNN.
    '''
    def __init__(self, d_input, hidden_size, d_output, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=d_input,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
        self.fc1 = nn.Linear(hidden_size, d_output)
    
    def forward(self, x):
        # (batch_size, num_steps, input_size)
        x
        # (batch_size, num_steps, hidden_size)
        h, _ = self.rnn(x, )
        # (batch_size, hidden_size)
        last_h = h[:, -1, :]
        # (batch_size, output_size)
        o = self.fc1(last_h)
        return o

class MLP(nn.Module):
    '''
    Multi-Layer Perception
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    '''
    LeNet-5
    use ReLU instead of Sigmoid to avoid the gradient explosion or vanishing
    when weights are inited not properly.
    '''
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 1 * (28 * 28)
        x
        # 6 * (28 * 28)
        x = torch.relu(self.conv1(x))
        # 6 * (14 * 14)
        x = self.maxpool1(x)
        # 16 * (10 * 10)
        x = torch.relu(self.conv2(x))
        # 16 * (5 * 5)
        x = self.maxpool2(x)
        # 16 * 5 * 5 = 400
        x = self.flatten(x)
        # 120
        x = torch.relu(self.fc1(x))
        # 84
        x = torch.relu(self.fc2(x))
        # 10
        x = self.fc3(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearSelfAttention(nn.Module):
    """
    Linear self-attention mechanism as described in the Linformer paper.
    """
    def __init__(self, embed_dim, num_heads, k_dim=None, dropout=0.1, 
                 share_kv=False, share_projection=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = math.sqrt(self.head_dim)
        
        # Projection dimension (k) - defaults to head_dim as in paper
        self.k_dim = k_dim if k_dim is not None else self.head_dim
        
        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Projection matrices E and F for K and V
        if share_projection:
            # Layerwise sharing - single projection matrix shared across all heads/layers
            self.E_proj = nn.Parameter(torch.Tensor(self.k_dim, embed_dim))
            self.F_proj = nn.Parameter(torch.Tensor(self.k_dim, embed_dim)) if not share_kv else self.E_proj
        else:
            # Headwise sharing - separate projections per head
            self.E_proj = nn.Parameter(torch.Tensor(num_heads, self.k_dim, self.head_dim))
            if share_kv:
                self.F_proj = self.E_proj  # Key-value sharing
            else:
                self.F_proj = nn.Parameter(torch.Tensor(num_heads, self.k_dim, self.head_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize projections like in original Transformer
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize E and F with random normal as suggested in paper
        if hasattr(self.E_proj, 'dim'):
            if self.E_proj.dim() == 2:  # Layerwise sharing
                nn.init.normal_(self.E_proj, mean=0, std=1/math.sqrt(self.k_dim))
                if not torch.equal(self.E_proj, self.F_proj):
                    nn.init.normal_(self.F_proj, mean=0, std=1/math.sqrt(self.k_dim))
            else:  # Headwise sharing
                nn.init.normal_(self.E_proj, mean=0, std=1/math.sqrt(self.k_dim))
                if not torch.equal(self.E_proj, self.F_proj):
                    nn.init.normal_(self.F_proj, mean=0, std=1/math.sqrt(self.k_dim))
        
        # Initialize biases
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: ByteTensor of shape (batch_size, seq_len) where 1 means padding
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Apply linear projections E and F to keys and values
        if hasattr(self.E_proj, 'dim') and self.E_proj.dim() == 3:  # Headwise sharing
            # Project keys: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, k_dim, head_dim)
            k = torch.einsum('bhnd,hkd->bhkn', k, self.E_proj)  # (batch_size, num_heads, k_dim, head_dim)
            # Project values: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, k_dim, head_dim)
            v = torch.einsum('bhnd,hkd->bhkn', v, self.F_proj)  # (batch_size, num_heads, k_dim, head_dim)
        else:  # Layerwise sharing
            # Project keys: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, k_dim, head_dim)
            k = k.transpose(1, 2).reshape(-1, seq_len, self.head_dim)  # (batch_size*num_heads, seq_len, head_dim)
            k = torch.einsum('bnd,kd->bnk', k, self.E_proj)  # (batch_size*num_heads, seq_len, k_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.k_dim).transpose(2, 3)  # (batch_size, num_heads, k_dim, seq_len)
            
            # Project values: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, k_dim, head_dim)
            v = v.transpose(1, 2).reshape(-1, seq_len, self.head_dim)  # (batch_size*num_heads, seq_len, head_dim)
            v = torch.einsum('bnd,kd->bnk', v, self.F_proj)  # (batch_size*num_heads, seq_len, k_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.k_dim).transpose(2, 3)  # (batch_size, num_heads, k_dim, seq_len)
        
        # Compute attention scores
        attn_scores = torch.matmul(q / self.scaling, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len, k_dim)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Expand mask to match attention scores shape
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads and project back to embedding dimension
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(context)
        
        return output

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network as in original Transformer.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class LinformerLayer(nn.Module):
    """
    A single Linformer layer consisting of linear self-attention and feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, 
                 k_dim=None, share_kv=False, share_projection=False):
        super().__init__()
        self.self_attn = LinearSelfAttention(
            embed_dim, num_heads, k_dim, dropout, share_kv, share_projection
        )
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class Linformer(nn.Module):
    """
    Linformer model for sequence prediction tasks.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, 
                 max_seq_len=512, dropout=0.1, k_dim=None, 
                 share_kv=False, share_projection=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Create Linformer layers
        self.layers = nn.ModuleList([
            LinformerLayer(
                embed_dim, num_heads, ff_dim, dropout, 
                k_dim, share_kv, share_projection
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.embed_dim**-0.5)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=self.embed_dim**-0.5)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.)
        
    def forward(self, input_ids, padding_mask=None):
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            padding_mask: ByteTensor of shape (batch_size, seq_len) where 1 means padding
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get token and position embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # Pass through each Linformer layer
        for layer in self.layers:
            x = layer(x, padding_mask)
        
        # Get output logits
        logits = self.output_layer(x)
        
        return logits

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    ff_dim = 2048
    max_seq_len = 512
    dropout = 0.1
    k_dim = 128  # Projection dimension (k) - can be much smaller than sequence length
    share_kv = True  # Share key and value projections
    share_projection = True  # Layerwise sharing of projection matrices
    
    # Create model
    model = Linformer(
        vocab_size, embed_dim, num_layers, num_heads, ff_dim,
        max_seq_len, dropout, k_dim, share_kv, share_projection
    )
    
    # Example input
    batch_size = 32
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # Example: no padding
    
    # Forward pass
    logits = model(input_ids, padding_mask)
    print("Output logits shape:", logits.shape)  # Should be (32, 128, 10000)
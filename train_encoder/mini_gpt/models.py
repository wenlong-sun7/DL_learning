"""
Neural network models for the Mini GPT Language Model
Contains MultiHeadSelfAttention, TransformerBlock, and Transformer classes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DROPOUT


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        # Compute scaled dot-product
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Apply masking if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return attn @ V

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project and split into heads
        Q = self.linear_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.linear_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.linear_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        attn_out = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""
    
    def __init__(self, embed_size, head_count):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, head_count)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attention_out = self.attention(x)
        x = self.norm1(attention_out + x)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        out = self.norm2(ff_out + x)
        
        return out


class Transformer(nn.Module):
    """Complete Transformer model for language modeling"""
    
    def __init__(self, vocab_size, embed_size, num_layers, head_counts):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # Embeddings
        self.word_embedding = nn.Embedding(vocab_size, embed_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, head_counts) 
            for _ in range(num_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def position_embedding(self, positions, embed_size):
        """Compute positional embeddings using sine/cosine functions"""
        # Compute angle rates for each position and dimension
        angle_rads = self.get_angles(
            positions.unsqueeze(2).float(),
            torch.arange(embed_size)[None, None, :].float().to(positions.device),
            embed_size
        )
        
        # Apply sine and cosine functions
        sines = torch.sin(angle_rads[:, :, 0::2])
        cosines = torch.cos(angle_rads[:, :, 1::2])
        pos_encoding = torch.cat([sines, cosines], dim=-1)
        pos_encoding = pos_encoding.squeeze(1)
        
        return pos_encoding.to(positions.device)

    def get_angles(self, pos, i, embed_size):
        """Compute angle rates for positional encoding"""
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / embed_size)
        return pos * angle_rates

    def forward(self, input_tokens, mask=None):
        batch_size, token_count = input_tokens.shape[:2]

        # Word embeddings
        out = self.word_embedding(input_tokens)

        # Positional embeddings
        positions = torch.arange(0, token_count).expand(batch_size, token_count).to(input_tokens.device)
        position_embedding = self.position_embedding(positions, self.embed_size)
        out = out + position_embedding.reshape(out.shape)

        # Pass through transformer layers
        for layer in self.layers:
            out = layer(out)

        # Output projection (use last token for language modeling)
        out = self.fc_out(out[:, -1, :].reshape(batch_size, self.embed_size))
        return out 
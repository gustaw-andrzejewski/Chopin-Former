import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """Head of a self-attention"""

    def __init__(self, embedding_dim: int, head_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        # Create a triangular mask with shape (max_seq_len, max_seq_len)
        self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.query(x)  # Compute query: shape [B, T, head_dim]
        K = self.key(x)  # Compute key: shape [B, T, head_dim]
        V = self.value(x)  # Compute value: shape [B, T, head_dim]

        # Scaled dot-product attention (using "@" for matrix multiplication)
        att_weights = Q @ K.transpose(-2, -1) / (self.head_dim**0.5)  # shape: [B, T, T]

        # Use the precomputed triangular mask and slice to current sequence length
        mask = self.tril[:T, :T]
        att_weights = att_weights.masked_fill(mask == 0, float("-inf"))
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)

        return att_weights @ V  # shape: [B, T, head_dim]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_relative_attention: bool = False,
        max_relative_distance: int = 16,
    ):
        super().__init__()
        if not use_relative_attention:
            self.heads = nn.ModuleList(
                [AttentionHead(embedding_dim, head_dim, max_seq_len, dropout) for _ in range(num_heads)]
            )
        else:
            self.heads = nn.ModuleList(
                [
                    RelativeAttentionHead(embedding_dim, head_dim, max_relative_distance, dropout, max_seq_len)
                    for _ in range(num_heads)
                ]
            )
        self.projection = nn.Linear(num_heads * head_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate outputs from each head along the last dimension.
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.projection(x)
        return x


class RelativeAttentionHead(nn.Module):
    """
    Attention head with relative positional encoding
    Inspired by: https://github.com/AliHaiderAhmad001/Self-Attention-with-Relative-Position-Representations
    """

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        max_relative_distance: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,  # Added max_seq_len parameter for mask size.
    ):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.relative_positional_encoding = RelativePositionalEncoding(head_dim, max_relative_distance)
        # Register a triangular mask of size (max_seq_len, max_seq_len)
        self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.query(x)  # [B, T, head_dim]
        K = self.key(x)  # [B, T, head_dim]
        V = self.value(x)  # [B, T, head_dim]

        # Generate relative bias matrices for keys and values.
        k_bias_matrix = self.relative_positional_encoding(
            T, T
        )  # shape: [T, T, head_dim] or [T, T] if implemented differently
        v_bias_matrix = self.relative_positional_encoding(T, T)

        # Compute self-attention scores.
        att_scores = Q @ K.transpose(-2, -1)
        # Compute relative attention scores.
        rel_att_scores = (Q.permute(1, 0, 2) @ k_bias_matrix.transpose(-2, -1)).transpose(0, 1)
        # Sum and scale.
        att_weights = (att_scores + rel_att_scores) / (self.head_dim**0.5)

        # Apply triangular mask.
        mask = self.tril[:T, :T]
        att_weights = att_weights.masked_fill(mask == 0, float("-inf"))
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)

        # Compute weighted sum for values.
        values = att_weights @ V
        # Compute relative representation for values.
        rel_values = torch.matmul(att_weights.permute(1, 0, 2), v_bias_matrix).transpose(0, 1)

        return values + rel_values


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding"""

    def __init__(self, relative_embedding_dim: int, max_relative_distance: int):
        super().__init__()
        self.max_relative_distance = max_relative_distance
        self.position_embeddings = nn.Parameter(
            torch.empty((2 * max_relative_distance + 1, relative_embedding_dim))
        )
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """Generate relative positional encoding"""
        query_indices = torch.arange(query_length)
        key_indices = torch.arange(key_length)
        distance_matrix = query_indices.unsqueeze(0) - key_indices.unsqueeze(1)
        distance_matrix = torch.clamp(distance_matrix, -self.max_relative_distance, self.max_relative_distance)
        distance_matrix = distance_matrix + self.max_relative_distance
        return self.position_embeddings[distance_matrix]

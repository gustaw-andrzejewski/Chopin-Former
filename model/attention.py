import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttentionHead(nn.Module):
    """Base class for attention heads with shared query, key, value projections and masking"""

    def __init__(self, embedding_dim: int, head_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = head_dim**0.5
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def _compute_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input tensor into query, key, and value representations"""
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        return Q, K, V

    def _apply_causal_mask(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply causal masking to ensure tokens only attend to previous positions"""
        mask_matrix = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill_(mask_matrix == 0, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        return self.dropout(attention_probs)


class AttentionHead(BaseAttentionHead):
    """Single attention head with scaled dot-product attention and causal masking"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input sequence using self-attention mechanism"""
        batch_size, seq_len, _ = x.shape
        Q, K, V = self._compute_qkv(x)

        attention_scores = Q @ K.transpose(-2, -1) / (self.scale_factor)

        attention_weights = self._apply_causal_mask(attention_scores, seq_len)

        return attention_weights @ V


class RelativeAttentionHead(BaseAttentionHead):
    """Attention head that incorporates relative positional information into the attention computation
    Inspired by: https://github.com/AliHaiderAhmad001/Self-Attention-with-Relative-Position-Representations"""

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        max_relative_distance: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__(embedding_dim, head_dim, max_seq_len, dropout)
        self.relative_positional_encoding = RelativePositionalEncoding(head_dim, max_relative_distance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input sequence using self-attention with relative positional encoding"""
        batch_size, seq_len, _ = x.shape
        Q, K, V = self._compute_qkv(x)

        pos_encoding = self.relative_positional_encoding(seq_len, seq_len)

        attention_scores = Q @ K.transpose(-2, -1)
        relative_attention_scores = torch.sum(Q.unsqueeze(2) * pos_encoding.unsqueeze(0), dim=-1)

        attention_weights = (attention_scores.add_(relative_attention_scores)) / (self.scale_factor)
        attention_weights = self._apply_causal_mask(attention_weights, seq_len)

        values = attention_weights @ V
        relative_values = torch.sum(attention_weights.unsqueeze(-1) * pos_encoding.unsqueeze(0), dim=2)

        return values + relative_values


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

        if use_relative_attention:
            self.heads = nn.ModuleList(
                [
                    RelativeAttentionHead(embedding_dim, head_dim, max_relative_distance, dropout, max_seq_len)
                    for _ in range(num_heads)
                ]
            )
        else:
            self.heads = nn.ModuleList(
                [AttentionHead(embedding_dim, head_dim, max_seq_len, dropout) for _ in range(num_heads)]
            )

        self.projection = nn.Linear(num_heads * head_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through attention heads and project their combined output"""
        head_outputs = torch.stack([head(x) for head in self.heads], dim=-2)

        B, T, num_heads, head_dim = head_outputs.shape
        x = head_outputs.reshape(B, T, num_heads * head_dim)

        x = self.projection(x)
        x = self.dropout(x)

        return x


class RelativePositionalEncoding(nn.Module):
    """Relative positional encodings for capturing token-pair relationships"""

    def __init__(self, relative_embedding_dim: int, max_relative_distance: int):
        super().__init__()
        self.max_relative_distance = max_relative_distance
        self.position_embeddings = nn.Parameter(
            torch.empty((2 * max_relative_distance + 1, relative_embedding_dim))
        )
        nn.init.xavier_uniform_(self.position_embeddings)
        self._cache = {}

    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """Generate distance-based position encodings for all query-key pairs"""
        cache_key = (query_length, key_length)

        if cache_key in self._cache:
            return self._cache[cache_key]

        device = self.position_embeddings.device
        query_positions = torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)

        relative_distances = query_positions.unsqueeze(0) - key_positions.unsqueeze(1)
        clamped_distances = torch.clamp(
            relative_distances, -self.max_relative_distance, self.max_relative_distance
        )
        embedding_indices = clamped_distances + self.max_relative_distance

        relative_embeddings = self.position_embeddings[embedding_indices]

        if len(self._cache) < 10:
            self._cache[cache_key] = relative_embeddings

        return relative_embeddings

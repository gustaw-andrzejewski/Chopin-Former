import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """Autoregressive Transformer Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(max_seq_len, embedding_dim, head_dim, num_heads) for _ in range(num_layers)]
        )
        self.projection = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x + self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.projection(x)
        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a sequence of tokens given a context."""
        generated_sequence = x
        for _ in range(self.max_seq_len):
            x = generated_sequence[:, -1:]
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs[:, -1], num_samples=1)
            generated_sequence = torch.cat([generated_sequence, next_token], dim=-1)
        return generated_sequence


class TransformerBlock(nn.Module):
    """A single block of a Transformer model."""

    def __init__(self, max_seq_len: int, embedding_dim: int, head_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(max_seq_len, embedding_dim, head_dim, num_heads)
        self.feed_forward = PointWiseFeedForward(embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network"""

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), nn.ReLU())
        self.projection = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.dropout(x)
        return self.projection(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, max_seq_len: int, embedding_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(max_seq_len, embedding_dim, head_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.projection(x)
        return x


class AttentionHead(nn.Module):
    """Head of a self-attention"""

    def __init__(self, max_seq_len: int, embedding_dim: int, head_dim: int):
        super().__init__()
        # Initializing query, key, and value linear transformations
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        # Creating a lower triangular mask to mask future positions
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)  # Compute query
        K = self.key(x)  # Compute key
        V = self.value(x)  # Compute value

        # Compute attention weights
        att_weights = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        att_weights = att_weights.masked_fill(self.mask == 0, float("-inf"))
        att_weights = F.softmax(att_weights, dim=-1)

        return att_weights @ V  # Apply attention weights to value


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, context_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("positional_encoding", self._get_positional_encoding(context_size, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_encoding[: x.shape[1], :]
        return x

    def _get_positional_encoding(self, context_size: int, embedding_dim: int) -> torch.Tensor:
        positional_encoding = torch.zeros(context_size, embedding_dim)
        for pos in range(context_size):
            for i in range(0, embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim)))
        return positional_encoding


if __name__ == "__main__":
    vocab_size = 10000  # Size of vocabulary
    max_seq_len = 512  # Maximum sequence length
    embedding_dim = 512  # Embedding dimension
    head_dim = 64  # Dimension of each attention head
    num_heads = 8  # Number of attention heads
    num_layers = 6  # Number of transformer layers (blocks)

    toy_input = torch.randint(0, vocab_size, (1, max_seq_len))  # Simulated batch of tokenized inputs
    toy_model = Transformer(vocab_size, embedding_dim, head_dim, num_heads, num_layers, max_seq_len)

    toy_output = toy_model(toy_input)  # Forward pass through the model
    print("Output shape:", toy_output.shape)  # Expected output shape: (batch_size, sequence_length, vocab_size)

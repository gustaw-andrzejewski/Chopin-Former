from typing import Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention


class Transformer(nn.Module):
    """Autoregressive Transformer Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        fcn_layer_size: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "leaky_relu"] = "gelu",
        use_relative_attention: bool = False,
        max_relative_distance: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    head_dim,
                    num_heads,
                    fcn_layer_size,
                    dropout,
                    activation,
                    max_seq_len,
                    use_relative_attention,
                    max_relative_distance,
                )
                for _ in range(num_layers)
            ]
        )
        self.projection = nn.Linear(embedding_dim, vocab_size)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.max_seq_len = max_seq_len
        self.projection.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Process input token sequence through the transformer model"""
        x = self.embedding(token_ids)
        x = self.positional_encoding(x)
        x = self.embed_dropout(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.final_norm(x)
        x = self.projection(x)
        return x

    def generate(self, token_sequence, max_length=256, temperature=1.0, top_k=50):
        """
        Autoregressively generate a sequence of tokens using the model.
        """
        for _ in range(max_length):
            context_sequence = token_sequence[:, -self.max_seq_len :]
            logits = self(context_sequence)[:, -1, :]
            logits = logits / temperature
            top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_indices, dim=-1, index=next_token_idx)
            token_sequence = torch.cat((token_sequence, next_token), dim=1)
        return token_sequence


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and feed-forward layers"""

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        num_heads: int,
        fcn_layer_size: int = 2048,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "leaky_relu"] = "gelu",
        max_seq_len: int = 512,
        use_relative_attention: bool = False,
        max_relative_distance: int = 16,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            embedding_dim,
            head_dim,
            num_heads,
            max_seq_len,
            dropout,
            use_relative_attention,
            max_relative_distance,
        )
        self.feed_forward = PointWiseFeedForward(embedding_dim, fcn_layer_size, dropout, activation)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feed-forward processing with residual connections"""
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network"""

    def __init__(
        self,
        embedding_dim: int,
        fcn_layer_size: int = 2048,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "leaky_relu"] = "gelu",
    ):
        super().__init__()
        self.activation = self._get_activation_function(activation)
        self.fc = nn.Sequential(nn.Linear(embedding_dim, fcn_layer_size), self.activation)
        self.projection = nn.Linear(fcn_layer_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input through feed-forward projection with activation and dropout"""
        x = self.fc(x)
        x = self.dropout(x)
        return self.projection(x)

    def _get_activation_function(self, activation: Literal["gelu", "relu", "leaky_relu"]):
        if activation == "gelu":
            return nn.GELU()
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unknown activation function: {activation}")


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        position_encodings = torch.zeros(max_seq_len, embedding_dim)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        frequency_factors = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim)
        )
        position_encodings[:, 0::2] = torch.sin(positions * frequency_factors)
        position_encodings[:, 1::2] = torch.cos(positions * frequency_factors)
        self.register_buffer("positional_encoding", position_encodings)
        self._cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to token embeddings"""
        seq_len = x.size(1)
        if seq_len in self._cache:
            return x + self._cache[seq_len]

        encoding = self.positional_encoding[:seq_len, :]

        if len(self._cache) < 10:
            self._cache[seq_len] = encoding

        return x + encoding


if __name__ == "__main__":
    from time import perf_counter

    vocab_size = 10000
    max_seq_len = 256
    embedding_dim = 256
    head_dim = 32
    num_heads = 4
    num_layers = 4
    use_relative_attention = True

    toy_input = torch.randint(0, vocab_size, (1, max_seq_len))
    print("Input shape:", toy_input.shape)
    toy_model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_relative_attention=use_relative_attention,
    )

    print("Starting toy model forward pass...")
    start = perf_counter()
    toy_output = toy_model(toy_input)
    print("Elapsed time:", perf_counter() - start)
    toy_generate = torch.zeros((1, 1), dtype=torch.long)

    print("Starting toy model generation...")
    start = perf_counter()
    toy_generate = toy_model.generate(toy_generate)
    print("Elapsed time:", perf_counter() - start)
    print("Generated sequence shape:", toy_generate.shape)

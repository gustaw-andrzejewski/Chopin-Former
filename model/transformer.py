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
        use_relative_attention: bool = False,
        max_relative_distance: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    head_dim,
                    num_heads,
                    fcn_layer_size,
                    dropout,
                    use_relative_attention,
                    max_relative_distance,
                    max_seq_len,  # Add max_seq_len parameter
                )
                for _ in range(num_layers)
            ]
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

    def generate(self, idx, max_length=256, temperature=1.0, top_k=50):
        """
        Generates a sequence of tokens using the model.

        Args:
            idx (torch.Tensor): The input sequence of tokens.
            max_length (int): The maximum length of the generated sequence. Defaults to 512.
            temperature (float): The temperature value for temperature scaling. Defaults to 1.0.
            top_k (int): The number of top-k tokens to consider during sampling. Defaults to 40.

        Returns:
            torch.Tensor: The generated sequence of tokens.
        """
        for _ in range(max_length):
            idx_cond = idx[:, -self.max_seq_len :]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.gather(top_indices, dim=-1, index=idx_next)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TransformerBlock(nn.Module):
    """A single block of a Transformer model."""

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        num_heads: int,
        fcn_layer_size: int = 2048,
        dropout: float = 0.1,
        use_relative_attention: bool = False,
        max_relative_distance: int = 16,
        max_seq_len: int = 512,  # Add max_seq_len parameter
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            embedding_dim,
            head_dim,
            num_heads,
            max_seq_len,  # Pass max_seq_len as the 4th parameter
            dropout,
            use_relative_attention,
            max_relative_distance,
        )
        self.feed_forward = PointWiseFeedForward(embedding_dim, fcn_layer_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network"""

    def __init__(self, embedding_dim: int, fcn_layer_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embedding_dim, fcn_layer_size), nn.ReLU())
        self.projection = nn.Linear(fcn_layer_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.dropout(x)
        return self.projection(x)


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Precompute the positional encodings and register as a buffer.
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, embedding_dim]
        T = x.size(1)
        return x + self.positional_encoding[:T, :]


if __name__ == "__main__":
    vocab_size = 10000  # Size of vocabulary
    max_seq_len = 256  # Maximum sequence length
    embedding_dim = 256  # Embedding dimension
    head_dim = 32  # Dimension of each attention head
    num_heads = 4  # Number of attention heads
    num_layers = 4  # Number of transformer layers (blocks)

    toy_input = torch.randint(0, vocab_size, (1, max_seq_len))  # Simulated batch of tokenized inputs
    toy_model = Transformer(vocab_size, embedding_dim, head_dim, num_heads, num_layers, max_seq_len)

    toy_output = toy_model(toy_input)  # Forward pass through the model
    print("Output shape:", toy_output.shape)  # Expected output shape: (batch_size, sequence_length, vocab_size)

    toy_generate = torch.zeros((1, 1), dtype=torch.long)  # Simulated BOS token
    toy_generate = toy_model.generate(toy_generate)  # Generate a sequence
    print("Generated sequence shape:", toy_generate.shape)  # Expected output shape: (batch_size, sequence_length)

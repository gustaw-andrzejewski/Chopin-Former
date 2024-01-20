import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy


class ChopinFormer(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        head_dim,
        num_heads,
        num_layers,
        fcn_layer_size,
        max_seq_len,
        dropout,
        learning_rate,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Transformer(
            vocab_size, embedding_dim, head_dim, num_heads, num_layers, fcn_layer_size, max_seq_len, dropout
        )
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=vocab_size)

    def forward(self, x):
        return self.transformer(x)

    def generate(self, x):
        return self.transformer.generate(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"][:, :-1]
        labels = batch["labels"][:, 1:]

        logits = self(input_ids)
        loss = self.loss(logits.reshape(-1, self.hparams.vocab_size), labels.reshape(-1))

        preds = torch.argmax(logits, dim=-1)
        acc = self.train_acc(preds.reshape(-1), labels.reshape(-1))
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"][:, :-1]
        labels = batch["labels"][:, 1:]

        logits = self(input_ids)
        loss = self.loss(logits.reshape(-1, self.hparams.vocab_size), labels.reshape(-1))

        preds = torch.argmax(logits, dim=-1)
        acc = self.val_acc(preds.reshape(-1), labels.reshape(-1))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}


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
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, head_dim, num_heads, fcn_layer_size, dropout)
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

    def generate(self, idx, max_length=512, temperature=1.0, top_k=40):
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
        self, embedding_dim: int, head_dim: int, num_heads: int, fcn_layer_size: int = 2048, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, head_dim, num_heads)
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


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embedding_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embedding_dim, head_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.projection(x)
        return x


class AttentionHead(nn.Module):
    """Head of a self-attention"""

    def __init__(self, embedding_dim: int, head_dim: int):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        self.head_dim = head_dim
        # Register buffer for the triangular mask
        self.register_buffer("tril", torch.tril(torch.ones(embedding_dim, embedding_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.query(x)  # Compute query
        K = self.key(x)  # Compute key
        V = self.value(x)  # Compute value

        att_weights = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Adjust the mask to the current sequence length
        mask = self.tril[:T, :T]

        att_weights = att_weights.masked_fill(mask == 0, float("-inf"))
        att_weights = F.softmax(att_weights, dim=-1)

        return att_weights @ V


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

    toy_generate = torch.zeros((1, 1), dtype=torch.long)  # Simulated BOS token
    toy_generate = toy_model.generate(toy_generate)  # Generate a sequence
    print("Generated sequence shape:", toy_generate.shape)  # Expected output shape: (batch_size, sequence_length)

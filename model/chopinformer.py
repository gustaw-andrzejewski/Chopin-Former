import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from model.transformer import Transformer


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
        weight_decay=0.005,
        use_relative_attention=False,
        max_relative_distance=16,
        activation="gelu",
        optimizer_name="adamw",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            fcn_layer_size=fcn_layer_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation=activation,
            use_relative_attention=use_relative_attention,
            max_relative_distance=max_relative_distance,
        )
        self.optimizer_name = optimizer_name
        self.optimizer_params = {"lr": learning_rate, "weight_decay": weight_decay}
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=vocab_size)

    def forward(self, x):
        return self.transformer(x)

    def generate(self, x, max_length: int = 256, temperature: float = 1.0, top_k: int = 50):
        return self.transformer.generate(x, max_length=max_length, temperature=temperature, top_k=top_k)

    def configure_optimizers(self):
        optimizer = self._get_optimizer(self.optimizer_name, self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _get_optimizer(self, optimizer_name, optimizer_params=None):
        if optimizer_params is None:
            optimizer_params = {}
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_params)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"][:, :-1]
        labels = batch["labels"][:, 1:]

        logits = self(input_ids)
        loss = self.loss(logits.reshape(-1, self.hparams.vocab_size), labels.reshape(-1))

        preds = torch.argmax(logits, dim=-1)
        acc = self.train_acc(preds.reshape(-1), labels.reshape(-1))
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False, logger=True)

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

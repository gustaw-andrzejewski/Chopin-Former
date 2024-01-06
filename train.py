from pathlib import Path

import lightning as L
import torch
from miditok import REMI
from miditok.pytorch_data import DataCollator, DatasetTok
from torch.utils.data import DataLoader
from torchtoolkit.data import create_subsets

from model import ChopinFormer


class MaestroDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_config: str,
        batch_size: int = 16,
        max_seq_len: int = 512,
        min_seq_len: int = 384,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_config = tokenizer_config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

    def setup(self, stage=None):
        self.tokenizer = REMI(params=Path(self.tokenizer_config))
        tokens_paths = list(Path(self.data_dir).glob("**/*.json"))
        dataset = DatasetTok(
            tokens_paths, max_seq_len=self.max_seq_len, min_seq_len=self.min_seq_len, one_token_stream=False
        )
        self.subset_train, self.subset_valid = create_subsets(dataset, [0.2])
        self.collator = DataCollator(
            self.tokenizer["PAD_None"],
            self.tokenizer["BOS_None"],
            self.tokenizer["EOS_None"],
            copy_inputs_as_labels=True,
        )

    def train_dataloader(self):
        return DataLoader(self.subset_train, num_workers=4, batch_size=self.batch_size, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.subset_valid, num_workers=4, batch_size=self.batch_size, collate_fn=self.collator)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    data_module = MaestroDataModule(
        data_dir="data/Maestro_tokens_bpe/",
        tokenizer_config="tokenizer/tokenizer_bpe.conf",
        batch_size=16,
        max_seq_len=510,
        min_seq_len=384,
    )

    model = ChopinFormer(
        vocab_size=10000,
        embedding_dim=512,
        head_dim=64,
        num_heads=8,
        num_layers=6,
        max_seq_len=512,
        dropout=0.2,
        learning_rate=1e-3,
    )

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=2,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(model, data_module)

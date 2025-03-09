import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from dataset import MaestroDataModule
from model import ChopinFormer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ChopinFormer on Maestro dataset")

    parser.add_argument(
        "--midi-dir",
        type=str,
        default="./dataset/data/maestro-v3.0.0",
        help="Directory containing raw MIDI files",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default="./dataset/tokenizer/tokenizer_bpe.conf",
        help="Path to tokenizer configuration file",
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="./dataset/data/maestro-chunks",
        help="Directory to store processed MIDI chunks (if None, uses temp dir)",
    )
    parser.add_argument(
        "--overlap-bars", type=int, default=2, help="Number of bars to overlap when chunking MIDIs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--min-seq-len", type=int, default=384, help="Minimum sequence length")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument(
        "--pre-tokenize", action="store_true", help="Pre-tokenize all data during setup for faster training"
    )

    parser.add_argument("--augment", action="store_true", help="Apply data augmentation to MIDI files")
    parser.add_argument(
        "--pitch-offsets",
        type=int,
        nargs="+",
        default=[-12, 12],
        help="Pitch offsets for augmentation (in semitones)",
    )
    parser.add_argument(
        "--velocity-offsets", type=int, nargs="+", default=[-4, 4], help="Velocity offsets for augmentation"
    )
    parser.add_argument(
        "--duration-offsets",
        type=float,
        nargs="+",
        default=[-0.5, 0.5],
        help="Duration offsets for augmentation (in beats)",
    )
    parser.add_argument(
        "--all-offset-combinations", action="store_true", help="Use all combinations of augmentation offsets"
    )

    parser.add_argument("--embedding-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--head-dim", type=int, default=32, help="Attention head dimension")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--fcn-layer-size", type=int, default=1536, help="Size of feed-forward network layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--relative-attention", action="store_true", help="Use relative attention")
    parser.add_argument(
        "--max-relative-distance", type=int, default=32, help="Maximum relative distance for relative attention"
    )
    parser.add_argument(
        "--activation", type=str, default="gelu", help="Activation function for feed-forward layers"
    )
    parser.add_argument("--optimizer-name", type=str, default="adamw", help="Optimizer name")

    parser.add_argument("--max-epochs", type=int, default=5, help="Maximum number of epochs")
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["16-mixed", "32", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument(
        "--project-name", type=str, default="ChopinFormer-Pretraining", help="Project name for wandb logging"
    )
    parser.add_argument("--run-name", type=str, default="piano-pretrain", help="Run name for wandb logging")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    midi_dir = Path(args.midi_dir).resolve()
    chunk_dir = Path(args.chunk_dir).resolve() if args.chunk_dir else None
    tokenizer_config = Path(args.tokenizer_config).resolve()

    data_module = MaestroDataModule(
        data_dir=midi_dir,
        tokenizer_config=tokenizer_config,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len - 1,
        min_seq_len=args.min_seq_len,
        num_workers=args.num_workers,
        chunk_dir=chunk_dir,
        overlap_bars=args.overlap_bars,
        pre_tokenize=args.pre_tokenize,
        augment=args.augment,
        pitch_offsets=args.pitch_offsets,
        velocity_offsets=args.velocity_offsets,
        duration_offsets=args.duration_offsets,
        all_offset_combinations=args.all_offset_combinations,
    )

    tokenizer = data_module.tokenizer
    vocab_size = tokenizer.vocab_size

    model = ChopinFormer(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        fcn_layer_size=args.fcn_layer_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_relative_attention=args.relative_attention,
        max_relative_distance=args.max_relative_distance,
        activation=args.activation,
        optimizer_name=args.optimizer_name,
    )

    logger = WandbLogger(project=args.project_name, name=args.run_name, log_model=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=args.checkpoint_dir,
        filename=f"{args.run_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        precision=args.precision,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        accumulate_grad_batches=2,
    )

    trainer.fit(model, data_module)

    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

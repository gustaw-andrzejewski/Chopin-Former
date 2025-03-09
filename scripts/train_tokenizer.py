"""
MIDI tokenizer training script for ChopinFormer.

This script handles training a tokenizer using MIDI files and various tokenization methods,
including BPE, Unigram, and WordPiece.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Literal

from miditok import REMI, TokenizerConfig

BASE_PATH = Path(__file__).parent.parent


def get_default_tokenizer_config() -> dict:
    """
    Get default tokenizer configuration.

    Returns:
        dictionary with tokenizer parameters
    """
    return {
        "pitch_range": (21, 109),
        "beat_res": {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1},
        "num_velocities": 24,
        "special_tokens": ["PAD", "BOS", "EOS"],
        "use_chords": False,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 32,
        "tempo_range": (50, 200),
    }


def initialize_tokenizer(config_path: Optional[Path] = None) -> REMI:
    """
    Initialize the REMI tokenizer with either a configuration file or default settings.
    """
    if config_path is not None and config_path.exists():
        print(f"Loading tokenizer from configuration: {config_path}")
        return REMI(params=config_path)

    print("Initializing tokenizer with default configuration")
    config = TokenizerConfig(**get_default_tokenizer_config())
    return REMI(config)


def train_tokenizer(
    midi_paths: list[Path],
    tokenizer_dir: Path,
    output_dir: Optional[Path] = None,
    vocab_size: int = 10000,
    model_type: Literal["BPE", "Unigram", "WordPiece"] = "BPE",
    tokenize_files: bool = False,
) -> Path:
    """
    Train a tokenizer on MIDI files and optionally tokenize them.
    """
    tokenizer_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = initialize_tokenizer()

    tokenizer_config_path = tokenizer_dir / f"tokenizer_{model_type.lower()}.conf"

    if not tokenizer_config_path.exists():
        print(f"Training tokenizer with vocab size {vocab_size} using {model_type} model")
        print(f"Processing {len(midi_paths)} MIDI files...")
        tokenizer.train(
            model=model_type,
            vocab_size=vocab_size,
            files_paths=midi_paths,
        )
        print(f"Saving tokenizer configuration to {tokenizer_config_path}")
        tokenizer.save_params(tokenizer_config_path)
    else:
        print(f"Loading existing tokenizer configuration from {tokenizer_config_path}")
        tokenizer = REMI(params=tokenizer_config_path)

    if tokenize_files and output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        tokens_dir = output_dir / f"tokens_{model_type.lower()}"
        tokens_dir.mkdir(exist_ok=True, parents=True)

        if not any(tokens_dir.iterdir()):
            print(f"Tokenizing {len(midi_paths)} MIDI files to {tokens_dir}")
            tokenizer.tokenize_dataset(midi_paths, tokens_dir)
        else:
            print(f"Tokens already exist at {tokens_dir}, skipping tokenization")

    return tokenizer_config_path


def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer for MIDI files.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing MIDI files to train tokenizer on",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=str(BASE_PATH / "dataset/tokenizer"),
        help="Directory to store tokenizer config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for tokenized outputs (if omitted, files won't be tokenized)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["BPE", "Unigram", "WordPiece"],
        default="BPE",
        help="Tokenization model type",
    )
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size for tokenization")
    parser.add_argument("--tokenize-files", action="store_true", help="Tokenize files after training")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)

    midi_paths = list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi"))
    if not midi_paths:
        print(f"No MIDI files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(midi_paths)} MIDI files for tokenizer training")

    tokenizer_config_path = train_tokenizer(
        midi_paths=midi_paths,
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        tokenize_files=args.tokenize_files,
    )

    print("\nTokenizer training complete!")
    print(f"Tokenizer configuration saved to {tokenizer_config_path}")

    print("\nTo use with train.py, use this path:")
    print(f'  --tokenizer-config="{tokenizer_config_path}"')


if __name__ == "__main__":
    main()

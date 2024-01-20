import subprocess
from pathlib import Path

from miditok import REMI, TokenizerConfig

BASE_PATH = Path(__file__).parent.parent


def download_dataset(dataset_dir):
    """Downloads and unzips the MIDI dataset."""
    if not list(dataset_dir.glob("**/*.mid")):
        subprocess.run(
            [
                "wget",
                "-P",
                str(dataset_dir),
                "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
            ]
        )
        subprocess.run(["unzip", str(dataset_dir / "maestro-v3.0.0-midi.zip"), "-d", str(dataset_dir)])
        subprocess.run(["rm", str(dataset_dir / "maestro-v3.0.0-midi.zip")])


def initialize_tokenizer():
    """Initializes the tokenizer with specified configuration."""
    config = TokenizerConfig(
        pitch_range=(21, 109),  # MIDI note range
        beat_res={(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1},
        num_velocities=24,
        special_tokens=["PAD", "BOS", "EOS"],
        use_tempos=True,
        num_tempos=32,
        tempo_range=(50, 200),
    )
    return REMI(config)


def tokenize_dataset(tokenizer, midi_paths, tokenized_data_dir):
    """Tokenizes the dataset and applies BPE."""
    tokenizer.tokenize_midi_dataset(midi_paths, tokenized_data_dir)
    tokenizer.learn_bpe(vocab_size=10000, tokens_paths=midi_paths, start_from_empty_voc=False)
    tokenizer.save_params(TOKENIZER_DIR / "tokenizer.json")


if __name__ == "__main__":
    DATASET_DIR = BASE_PATH / Path("dataset/Maestro")
    TOKENIZER_DIR = BASE_PATH / Path("tokenizer")
    TOKENIZED_DATA_DIR = BASE_PATH / Path("dataset/Maestro_tokens")

    # Ensure directories exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    download_dataset(DATASET_DIR)

    # Initialize the tokenizer
    tokenizer = initialize_tokenizer()

    # Get MIDI paths and tokenize the dataset
    midi_paths = list(DATASET_DIR.glob("**/*.mid")) + list(DATASET_DIR.glob("**/*.midi"))
    tokenize_dataset(tokenizer, midi_paths, TOKENIZED_DATA_DIR)

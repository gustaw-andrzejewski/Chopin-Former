from pathlib import Path
import tempfile
import shutil
import random

import lightning as L
from miditok import REMI
from miditok.pytorch_data import DataCollator, DatasetMIDI

from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset
from torch.utils.data import DataLoader


class MaestroDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_config: str,
        batch_size: int = 16,
        max_seq_len: int = 512,
        min_seq_len: int = 384,
        val_split: float = 0.1,
        num_workers: int = 6,
        chunk_dir: str = None,
        overlap_bars: int = 2,
        pre_tokenize: bool = False,
        augment: bool = False,
        pitch_offsets: list = None,
        velocity_offsets: list = None,
        duration_offsets: list = None,
        all_offset_combinations: bool = False,
    ):
        """
        MIDI Dataset with on-the-fly tokenization and data augmentation

        Args:
            data_dir: Directory containing MIDI files
            tokenizer_config: Path to saved tokenizer configuration
            batch_size: Batch size for training
            max_seq_len: Maximum sequence length in tokens
            min_seq_len: Minimum sequence length in tokens
            val_split: Fraction of data to use for validation
            num_workers: Number of workers for data loading
            chunk_dir: Directory to save chunked MIDI files (if None, uses temp dir)
            overlap_bars: Number of bars to overlap when chunking
            pre_tokenize: Whether to tokenize all data during setup
            augment: Whether to perform data augmentation
            pitch_offsets: List of pitch offsets for augmentation (default: [-12, 12])
            velocity_offsets: List of velocity offsets for augmentation (default: [-4, 4])
            duration_offsets: List of duration offsets in beats (default: [-0.5, 0.5])
            all_offset_combinations: Whether to use all combinations of offsets
        """
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_config = tokenizer_config
        self.tokenizer = REMI(params=Path(tokenizer_config))
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.val_split = val_split
        self.num_workers = num_workers
        self.chunk_dir = chunk_dir
        self.overlap_bars = overlap_bars
        self.pre_tokenize = pre_tokenize

        # Data augmentation parameters
        self.augment = augment
        self.pitch_offsets = pitch_offsets if pitch_offsets is not None else [-12, 12]
        self.velocity_offsets = velocity_offsets if velocity_offsets is not None else [-4, 4]
        self.duration_offsets = duration_offsets if duration_offsets is not None else [-0.5, 0.5]
        self.all_offset_combinations = all_offset_combinations

        # Will be set up during setup()
        self.base_dir = None
        self.processed_dir = None
        self.train_dir = None
        self.valid_dir = None
        self.is_temp_dir = False
        self.dataset_train = None
        self.dataset_valid = None
        self.collator = None
        self.vocab_size = None

    def prepare_data(self):
        """
        Prepare data in this order:
        1. Augment the dataset if requested
        2. Split into train/validation
        3. Chunk each split for training

        This runs once on the main process.
        """

        if self.chunk_dir is None:
            self.base_dir = Path(tempfile.mkdtemp())
            self.is_temp_dir = True
        else:
            self.base_dir = Path(self.chunk_dir)
            self.base_dir.mkdir(exist_ok=True, parents=True)

        midi_paths = list(Path(self.data_dir).glob("**/*.mid")) + list(Path(self.data_dir).glob("**/*.midi"))
        if not midi_paths:
            raise ValueError(f"No MIDI files found in {self.data_dir}. Please check the path.")
        print(f"Found {len(midi_paths)} MIDI files for processing")

        if self.augment:
            print("Performing data augmentation:")
            print(f"  Pitch offsets: {self.pitch_offsets}")
            print(f"  Velocity offsets: {self.velocity_offsets}")
            print(f"  Duration offsets: {self.duration_offsets}")
            print(f"  Using all combinations: {self.all_offset_combinations}")

            aug_dir = self.base_dir / "augmented"
            aug_dir.mkdir(exist_ok=True, parents=True)

            augment_dataset(
                data_path=self.data_dir,
                pitch_offsets=self.pitch_offsets,
                velocity_offsets=self.velocity_offsets,
                duration_offsets=self.duration_offsets,
                all_offset_combinations=self.all_offset_combinations,
                out_path=aug_dir,
            )

            all_midi_paths = list(aug_dir.glob("**/*.mid")) + list(aug_dir.glob("**/*.midi"))
            print(f"After augmentation: {len(all_midi_paths)} MIDI files available")
        else:
            all_midi_paths = midi_paths

        random.shuffle(all_midi_paths)
        num_valid = max(1, int(len(all_midi_paths) * self.val_split))
        midi_paths_valid = all_midi_paths[:num_valid]
        midi_paths_train = all_midi_paths[num_valid:]

        print(f"Split dataset: {len(midi_paths_train)} training files, {len(midi_paths_valid)} validation files")

        self.train_dir = self.base_dir / "train_chunks"
        self.valid_dir = self.base_dir / "valid_chunks"
        self.train_dir.mkdir(exist_ok=True, parents=True)
        self.valid_dir.mkdir(exist_ok=True, parents=True)

        print(f"Chunking training files into {self.train_dir}")
        split_files_for_training(
            files_paths=midi_paths_train,
            tokenizer=self.tokenizer,
            save_dir=self.train_dir,
            max_seq_len=self.max_seq_len,
            num_overlap_bars=self.overlap_bars,
        )

        print(f"Chunking validation files into {self.valid_dir}")
        split_files_for_training(
            files_paths=midi_paths_valid,
            tokenizer=self.tokenizer,
            save_dir=self.valid_dir,
            max_seq_len=self.max_seq_len,
            num_overlap_bars=self.overlap_bars,
        )

        train_chunks = list(self.train_dir.glob("**/*.mid")) + list(self.train_dir.glob("**/*.midi"))
        valid_chunks = list(self.valid_dir.glob("**/*.mid")) + list(self.valid_dir.glob("**/*.midi"))
        print(f"Created {len(train_chunks)} training chunks and {len(valid_chunks)} validation chunks")

    def setup(self, stage=None):
        """
        Set up datasets and tokenizer. This runs on every GPU.
        """
        if self.tokenizer is None:
            self.tokenizer = REMI(params=Path(self.tokenizer_config))

        if self.train_dir is None or self.valid_dir is None:
            if self.chunk_dir is None:
                raise RuntimeError("MIDI chunks directories not set. Did you call prepare_data()?")
            self.train_dir = Path(self.chunk_dir) / "train_chunks"
            self.valid_dir = Path(self.chunk_dir) / "valid_chunks"

        train_chunks = list(self.train_dir.glob("**/*.mid")) + list(self.train_dir.glob("**/*.midi"))
        valid_chunks = list(self.valid_dir.glob("**/*.mid")) + list(self.valid_dir.glob("**/*.midi"))

        if not train_chunks or not valid_chunks:
            raise ValueError("Missing MIDI chunks in training or validation directories")

        print(f"Found {len(train_chunks)} training chunks and {len(valid_chunks)} validation chunks")

        self.dataset_train = DatasetMIDI(
            files_paths=train_chunks,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            bos_token_id=self.tokenizer["BOS_None"],
            eos_token_id=self.tokenizer["EOS_None"],
            pre_tokenize=self.pre_tokenize,
        )

        self.dataset_valid = DatasetMIDI(
            files_paths=valid_chunks,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            bos_token_id=self.tokenizer["BOS_None"],
            eos_token_id=self.tokenizer["EOS_None"],
            pre_tokenize=self.pre_tokenize,
        )

        print(
            f"Created datasets with {len(self.dataset_train)} training samples and {len(self.dataset_valid)} validation samples"
        )

        self.collator = DataCollator(pad_token_id=self.tokenizer["PAD_None"], copy_inputs_as_labels=True)

        self.vocab_size = self.tokenizer.vocab_size
        print(f"Vocabulary size: {self.vocab_size}")

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collator
        )

    def teardown(self, stage=None):
        """
        Clean up temporary directories if used
        """
        if self.is_temp_dir and self.base_dir and self.base_dir.exists():
            print(f"Cleaning up temporary directory {self.base_dir}")
            shutil.rmtree(self.base_dir)


if __name__ == "__main__":
    data_module = MaestroDataModule(
        data_dir=Path("data/maestro-v3.0.0"),
        tokenizer_config="tokenizer/tokenizer_bpe.conf",
        batch_size=2,
        max_seq_len=256,
        min_seq_len=128,
        val_split=0.1,
        num_workers=6,
        chunk_dir=None,
        overlap_bars=2,
        pre_tokenize=True,
        augment=True,
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
        all_offset_combinations=True,
    )
    data_module.prepare_data()

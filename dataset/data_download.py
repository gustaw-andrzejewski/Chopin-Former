"""
Maestro MIDI dataset downloader.

This script downloads the Maestro MIDI dataset.
"""

import argparse
from pathlib import Path
import zipfile
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_maestro(data_dir: Path, version: str = "v3.0.0") -> Path:
    """
    Download the Maestro MIDI dataset if not already present.
    """
    dataset_path = data_dir / f"maestro-{version}"
    zip_path = data_dir / f"maestro-{version}-midi.zip"

    if dataset_path.exists():
        print(f"Maestro dataset already exists at {dataset_path}")
        return dataset_path

    data_dir.mkdir(exist_ok=True, parents=True)

    url = f"https://storage.googleapis.com/magentadata/datasets/maestro/{version}/maestro-{version}-midi.zip"
    print(f"Downloading Maestro MIDI dataset from {url}")

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Maestro") as t:
        urlretrieve(url, filename=zip_path, reporthook=t.update_to)

    print(f"Extracting dataset to {data_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    zip_path.unlink()

    return dataset_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download the Maestro dataset for ChopinFormer.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store the dataset")
    parser.add_argument("--version", type=str, default="v3.0.0", help="Maestro dataset version")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    download_maestro(data_dir, args.version)


if __name__ == "__main__":
    main()

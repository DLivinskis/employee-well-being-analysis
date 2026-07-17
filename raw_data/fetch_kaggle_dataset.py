"""
Fetches the "remote-work-and-mental-health" Kaggle dataset.

Downloads the dataset via kagglehub (which caches it under
~/.cache/kagglehub) and copies the resulting files into this script's
directory, so raw_data/ ends up with a local copy alongside the fetch
script.
"""

import shutil
from pathlib import Path

import kagglehub


def fetch_dataset(handle: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset and copy its files into dest_dir.

    Args:
        handle: Kaggle dataset identifier, e.g. "owner/dataset-name".
        dest_dir: Directory to copy the downloaded files into.

    Returns:
        The dest_dir path, once all files have been copied into it.

    Raises:
        OSError: If a downloaded file cannot be copied to dest_dir.
    """
    cache_path = Path(kagglehub.dataset_download(handle))
    for file in cache_path.iterdir():
        shutil.copy2(file, dest_dir / file.name)
    return dest_dir


if __name__ == "__main__":
    result_dir = fetch_dataset(
        "waqi786/remote-work-and-mental-health", Path(__file__).parent
    )
    print("Path to dataset files:", result_dir)

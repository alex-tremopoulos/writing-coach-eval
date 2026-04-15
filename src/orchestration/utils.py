"""Utility functions for pipeline orchestration."""

from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    """Return a timestamp string suitable for folder naming (MMDD_HHMM)."""
    return datetime.now().strftime("%m%d_%H%M")


def find_file_ending_with(directory: Path, suffix: str) -> Path:
    """Return the newest file in *directory* whose name ends with *suffix*.

    Args:
        directory: Directory to search.
        suffix: File suffix to match (e.g. "_enriched.csv").

    Returns:
        Path to the newest matching file.

    Raises:
        FileNotFoundError: If no file matching the suffix is found.
    """
    candidates = sorted(directory.glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No file ending with '{suffix}' found in {directory}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


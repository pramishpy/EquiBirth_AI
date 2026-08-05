"""Small filesystem helpers shared across the pipeline.

Kept deliberately thin. Heavier data IO (parquet read/write, integrity checks) lives in
the ingest package where it has access to the data engineering dependencies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks (safe for large files)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

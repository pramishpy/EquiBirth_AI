"""Download natality CSVs from the NBER mirror and record integrity metadata.

The full per-year CSV is large (roughly 2.5 GB). For development and CI we support a byte-range
sample download that is trimmed to the last complete line so it parses cleanly. The manifest records
a SHA-256, byte size, row count, source URL, and download date per file so a silently revised remote
file is detected (NFR6).
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.request
from pathlib import Path

from equibirth.utils.io import ensure_dir, sha256sum

CSV_URL_TEMPLATE = "https://data.nber.org/nvss/natality/csv/{year}/natality{year}us.csv"


def build_url(year: int, template: str = CSV_URL_TEMPLATE) -> str:
    """Return the NBER CSV URL for a given year."""
    return template.format(year=year)


def _trim_to_last_newline(data: bytes) -> bytes:
    """Drop a trailing partial line (range downloads cut mid-row)."""
    cut = data.rfind(b"\n")
    return data[:cut] if cut != -1 else data


def download_csv(
    year: int,
    out_dir: str | Path,
    *,
    max_bytes: int | None = None,
    template: str = CSV_URL_TEMPLATE,
    timeout: int = 600,
) -> Path:
    """Download a year's CSV to ``out_dir``.

    ``max_bytes`` requests only the first N bytes via an HTTP Range header (a representativeness-free
    sample for code validation; see load.load_natality_csv caveat). Without it, the full file is
    downloaded. Returns the written path.
    """
    out_dir = ensure_dir(out_dir)
    url = build_url(year, template)
    suffix = "_sample" if max_bytes else ""
    out_path = out_dir / f"natality{year}{suffix}.csv"

    req = urllib.request.Request(url)
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if max_bytes:
        data = _trim_to_last_newline(data)
    out_path.write_bytes(data)
    return out_path


def _http_range(url: str, start: int, size: int, timeout: int = 300) -> bytes:
    """Fetch ``size`` bytes from ``start`` via an HTTP Range request."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def remote_size(url: str, timeout: int = 60) -> int:
    """Total byte size of a remote file via a 1-byte Range request's Content-Range header."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cr = resp.headers.get("Content-Range", "")
    if "/" not in cr:
        raise RuntimeError(f"Server did not return Content-Range for {url}")
    return int(cr.rsplit("/", 1)[1])


def assemble_spread_sample(
    chunks: list[bytes], *, has_header_first: bool = True
) -> bytes:
    """Assemble byte chunks taken from spaced offsets into one valid CSV.

    The first chunk keeps its header; every chunk is trimmed to whole lines (a range request cuts
    mid-row at both ends). Pure function so the offset/trim logic is unit-testable without network.
    """
    parts: list[bytes] = []
    for i, raw in enumerate(chunks):
        body = raw
        if i == 0 and has_header_first:
            # keep header + whole rows; discard a partial tail (or a fragment with no newline at all)
            body = body[: body.rfind(b"\n")] if b"\n" in body else b""
        else:
            # drop partial first line (mid-row start) and partial last line
            start = body.find(b"\n")
            body = body[start + 1:] if start != -1 else b""
            end = body.rfind(b"\n")
            body = body[:end] if end != -1 else b""
        if body:
            parts.append(body)
    return b"\n".join(parts)


def build_spread_sample(
    year: int,
    out_dir: str | Path,
    *,
    n_chunks: int = 10,
    chunk_bytes: int = 18_000_000,
    template: str = CSV_URL_TEMPLATE,
    fetch=None,
    total_bytes: int | None = None,
) -> Path:
    """Download a representative spread sample: ``n_chunks`` chunks at evenly spaced byte offsets.

    A head slice of the (sorted) NBER CSV is biased; sampling across the whole file is representative
    (validated in FINDINGS F2). ``fetch(url, start, size)`` and ``total_bytes`` are injectable for
    tests. Returns the written path.
    """
    out_dir = ensure_dir(out_dir)
    url = build_url(year, template)
    fetch = fetch or _http_range
    total = total_bytes if total_bytes is not None else remote_size(url)
    step = total // n_chunks
    chunks = [fetch(url, i * step, chunk_bytes) for i in range(n_chunks)]
    data = assemble_spread_sample(chunks)
    out_path = out_dir / f"natality{year}_spread.csv"
    out_path.write_bytes(data)
    return out_path


def count_csv_rows(path: str | Path) -> int:
    """Count data rows (excluding the header) without loading the file into memory."""
    path = Path(path)
    with path.open("rb") as f:
        total = sum(1 for _ in f)
    return max(total - 1, 0)


def write_manifest_entry(
    manifest_path: str | Path,
    path: str | Path,
    *,
    year: int,
    url: str,
    is_sample: bool,
) -> dict:
    """Add or update a file's integrity record in the JSON manifest and return the entry."""
    manifest_path = Path(manifest_path)
    path = Path(path)
    entry = {
        "year": year,
        "file": path.name,
        "url": url,
        "is_sample": is_sample,
        "sha256": sha256sum(path),
        "bytes": path.stat().st_size,
        "rows": count_csv_rows(path),
        "downloaded": _dt.date.today().isoformat(),
    }
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[path.name] = entry
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return entry


def verify_manifest(manifest_path: str | Path, data_dir: str | Path) -> list[str]:
    """Re-hash files referenced by the manifest; return a list of mismatch messages (empty = ok)."""
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    problems: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        fpath = data_dir / name
        if not fpath.is_file():
            problems.append(f"{name}: missing on disk")
            continue
        if sha256sum(fpath) != entry["sha256"]:
            problems.append(f"{name}: sha256 mismatch")
    return problems

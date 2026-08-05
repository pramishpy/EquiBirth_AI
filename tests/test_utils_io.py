"""Tests for filesystem helpers."""

from equibirth.utils.io import ensure_dir, sha256sum


def test_ensure_dir_creates_nested_path(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.is_dir()
    # Idempotent: calling again does not raise.
    ensure_dir(target)


def test_sha256sum_is_stable(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"equibirth")
    digest = sha256sum(f)
    assert digest == sha256sum(f)
    assert len(digest) == 64

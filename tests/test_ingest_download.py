"""Tests for download URL building and manifest integrity (no network required)."""

from equibirth.ingest import download


def test_build_url():
    assert download.build_url(2024) == "https://data.nber.org/nvss/natality/csv/2024/natality2024us.csv"


def test_trim_to_last_newline():
    assert download._trim_to_last_newline(b"a\nb\nc") == b"a\nb"
    assert download._trim_to_last_newline(b"a\nb\n") == b"a\nb"


def test_count_csv_rows(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("header\n1\n2\n3\n", encoding="utf-8")
    assert download.count_csv_rows(f) == 3


def test_assemble_spread_sample_trims_partial_lines():
    # chunk 0 keeps header, drops partial tail; later chunks drop partial head and tail.
    c0 = b"header\nrow1\nrow2\npar"           # 'par' partial tail dropped
    c1 = b"tial\nrowA\nrowB\npar2"           # 'tial' partial head and 'par2' tail dropped
    out = download.assemble_spread_sample([c0, c1])
    assert out == b"header\nrow1\nrow2\nrowA\nrowB"


def test_build_spread_sample_uses_injected_fetch(tmp_path):
    calls = []

    def fake_fetch(url, start, size):
        calls.append(start)
        # each chunk: a partial head (except chunk0 has header) + whole rows + partial tail
        if start == 0:
            return b"h\n1\n2\nXX"
        return b"YY\n3\n4\nZZ"

    out = download.build_spread_sample(
        2099, tmp_path, n_chunks=2, chunk_bytes=10, total_bytes=1000, fetch=fake_fetch
    )
    assert calls == [0, 500]                      # evenly spaced offsets
    text = out.read_text()
    assert text == "h\n1\n2\n3\n4"               # header once, partials trimmed


def test_manifest_write_and_verify_roundtrip(tmp_path):
    data = tmp_path / "natality2099_sample.csv"
    data.write_text("header\n1\n2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    entry = download.write_manifest_entry(
        manifest, data, year=2099, url="http://x", is_sample=True
    )
    assert entry["rows"] == 2
    assert entry["year"] == 2099
    assert download.verify_manifest(manifest, tmp_path) == []

    # Corrupt the file -> verify reports a mismatch.
    data.write_text("header\n1\n2\n3\n", encoding="utf-8")
    problems = download.verify_manifest(manifest, tmp_path)
    assert any("sha256 mismatch" in p for p in problems)

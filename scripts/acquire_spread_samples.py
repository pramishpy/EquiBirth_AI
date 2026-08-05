"""SAMPLE-ERA (retained): part of the spread-sample chain that regenerates the Streamlit app's
reports/dashboard_data.json bundle. Its numbers are NOT the canonical ones and must not be cited;
the canonical full-data (2018-2024) pipeline lives in scripts/week*_*.py and reports/week*. This
chain is kept only so the app's exploration sections stay reproducible, and the app discloses on
screen that those sections use a validated representative spread sample.

Acquire representative spread samples for every natality year (2019-2024).

Each full-year CSV is ~2.2 GB and the link throttles, so a full pull times out. A spread sample
(chunks at evenly spaced byte offsets) is representative (FINDINGS F2). This builds one per year,
records a manifest entry, and prints row counts. Run in the background; it is bandwidth-bound.

Run: ./.venv/bin/python scripts/acquire_spread_samples.py
"""

from __future__ import annotations

from pathlib import Path

from equibirth.ingest import download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def main() -> None:
    for year in YEARS:
        out = RAW / f"natality{year}_spread.csv"
        if out.is_file():
            print(f"{year}: already present, skipping")
        else:
            print(f"{year}: downloading spread sample...", flush=True)
            out = download.build_spread_sample(year, RAW)
        entry = download.write_manifest_entry(
            MANIFEST, out, year=year, url=download.build_url(year), is_sample=True
        )
        print(f"{year}: rows={entry['rows']:,} bytes={entry['bytes']:,}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

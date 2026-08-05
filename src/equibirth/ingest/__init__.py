"""Data ingestion: download, schema and leakage classification, cleaning, Parquet conversion.

Modules:
- ``schema``: raw column names, harmonized mapping, and the pre- vs post-delivery leakage split.
- ``download``: fetch natality CSVs from the NBER mirror with a checksum/row-count manifest.
- ``load``: parse one year into a clean Polars frame (per-variable sentinels, target, race label).
- ``to_parquet``: memory-safe lazy streaming CSV to slim cleaned Parquet.
- ``confirm``: the FR1.1 per-year format-confirmation gate and its audit trail.
"""

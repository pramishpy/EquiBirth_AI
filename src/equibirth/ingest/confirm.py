"""FR1.1 format-confirmation gate: machine-written audit trail.

The human-written DECISION_LOG records the conclusion; this module produces a machine record so a
reviewer can verify programmatically that each year's CSV column layout was checked (target present,
LMP variable present and excluded, key features present) before any modeling used it.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from equibirth.ingest import schema
from equibirth.utils.io import ensure_dir


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing (FR1.4)."""


# Minimal set of raw columns every year's CSV must expose for the pipeline to be valid.
REQUIRED_COLUMNS: tuple[str, ...] = (
    schema.OBSTETRIC_GESTATION, schema.LMP_GESTATION, "dplural",
    "mracehisp", "meduc", "pay", "wic", "rf_ppterm", "mager",
)


def confirm_format(year: int, columns: list[str]) -> dict:
    """Validate a year's column set and return a confirmation record.

    Raises ``ConfigurationError`` if any required column is absent, so a layout drift fails loudly
    rather than silently producing a wrong target or missing feature.
    """
    cols = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ConfigurationError(f"{year}: required columns missing from CSV: {missing}")
    return {
        "year": year,
        "n_columns": len(columns),
        "target_column": schema.OBSTETRIC_GESTATION,
        "lmp_excluded_column": schema.LMP_GESTATION,
        "format": "csv",
        "confirmed": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def log_format_confirmation(record: dict, path: str | Path) -> Path:
    """Append or update a year's confirmation record in a JSON log keyed by year."""
    path = Path(path)
    ensure_dir(path.parent)
    log: dict = {}
    if path.is_file():
        log = json.loads(path.read_text(encoding="utf-8"))
    log[str(record["year"])] = record
    path.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_gestational_maps(cfg) -> None:
    """FR1.4 startup guard: the per-year gestational-age maps must be populated."""
    if not cfg.get("data.gestational_age_var_by_year"):
        raise ConfigurationError("config.data.gestational_age_var_by_year is empty (FR1.4).")
    if not cfg.get("data.lmp_gestation_var_by_year"):
        raise ConfigurationError("config.data.lmp_gestation_var_by_year is empty (FR1.4).")

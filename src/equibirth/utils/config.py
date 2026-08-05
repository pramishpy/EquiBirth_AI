"""Load and access the project configuration.

A single ``config.yaml`` is the one source of runtime settings (paths, seeds, sample
sizes, model params). Nothing else in the codebase should hardcode these values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Read-only wrapper over the parsed config with dotted-path access.

    Behaves like a mapping (``cfg["project"]["random_seed"]``) and also supports
    ``cfg.get("project.random_seed")`` for convenience and safe defaults.
    """

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return the value at a dotted path, or ``default`` if any segment is missing."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the underlying dict."""
        return dict(self._data)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Parse ``path`` (YAML) into a :class:`Config`.

    Raises ``FileNotFoundError`` if the file does not exist so misconfiguration fails
    loudly rather than silently using defaults.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return Config(data, source=path)

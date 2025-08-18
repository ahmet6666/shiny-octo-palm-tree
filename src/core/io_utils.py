from __future__ import annotations

"""I/O helper utilities."""

from pathlib import Path
from typing import Any

import json
import pandas as pd


def ensure_parent(path: Path) -> None:
    """Ensure that the parent directory of ``path`` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write ``df`` to ``path`` in Parquet format."""
    ensure_parent(path)
    df.to_parquet(path, **kwargs)


def read_json(path: Path) -> Any:
    """Read JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    """Write JSON file."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

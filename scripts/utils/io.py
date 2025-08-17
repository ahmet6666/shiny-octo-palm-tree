"""Utility functions for IO operations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class Paths:
    base_dir: Path
    raw_dir: Path
    processed_dir: Path
    raw_file: Path
    processed_file: Path
    meta_file: Path


def get_paths(config: Dict[str, Any]) -> Paths:
    """Return structured paths based on configuration."""
    data_cfg = config["data"]
    base_dir = Path(data_cfg["base_dir"]).resolve()
    raw_dir = Path(data_cfg["raw_dir"]).resolve()
    processed_dir = Path(data_cfg["processed_dir"]).resolve()
    raw_file = raw_dir / data_cfg["raw_filename"]
    processed_file = processed_dir / data_cfg["processed_filename"]
    meta_file = processed_dir / data_cfg["meta_filename"]
    return Paths(base_dir, raw_dir, processed_dir, raw_file, processed_file, meta_file)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    df.to_parquet(path, index=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

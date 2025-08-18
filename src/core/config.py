from __future__ import annotations

"""Configuration schema and loader."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel
import yaml


class DateRange(BaseModel):
    start: str
    end: str


class Paths(BaseModel):
    root: Path = Path(".")
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    reports_dir: Path


class Config(BaseModel):
    symbol: str = "BTCUSDT"
    market: str = "spot"
    timeframe: str = "1h"
    date_range: DateRange
    paths: Paths
    data_check: dict
    preprocess: dict
    technical_indicators: dict
    target: dict
    split: dict
    scaling: dict
    train: dict
    metric: dict
    lstm_baseline: dict
    tracking: dict
    ui: dict


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # resolve paths relative to config file
    base = path.parent
    data["paths"]["data_dir"] = base / data["paths"]["data_dir"]
    data["paths"]["raw_dir"] = base / data["paths"]["raw_dir"]
    data["paths"]["processed_dir"] = base / data["paths"]["processed_dir"]
    data["paths"]["artifacts_dir"] = base / data["paths"]["artifacts_dir"]
    data["paths"]["logs_dir"] = base / data["paths"]["logs_dir"]
    data["paths"]["reports_dir"] = base / data["paths"]["reports_dir"]
    return Config(**data)

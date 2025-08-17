"""Check availability and integrity of local raw data."""
from __future__ import annotations

import argparse
import logging

import pandas as pd
import pytz
import yaml

from utils.io import get_paths, read_parquet
from utils.logging_setup import setup_logging


logger = logging.getLogger(__name__)


def check_local_data(config: dict) -> dict:
    paths = get_paths(config)
    raw_path = paths.raw_file
    if not raw_path.exists():
        return {"exists": False, "reason": "raw file missing"}

    df = read_parquet(raw_path)
    if "open_time" not in df.columns:
        return {"exists": False, "reason": "open_time column missing"}

    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_index()
    reindexed = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="1H", tz=pytz.UTC))
    if reindexed.isnull().any().any():
        return {"exists": False, "reason": "missing candles detected"}

    if df.index.tz != pytz.UTC:
        return {"exists": False, "reason": "timezone not UTC"}

    return {"exists": True, "reason": "ok"}


def main(cfg_path: str):
    setup_logging()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    result = check_local_data(config)
    logger.info("Data check result: %s", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

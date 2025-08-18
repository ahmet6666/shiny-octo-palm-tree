from __future__ import annotations

"""Preprocess raw kline data."""

from pathlib import Path
import pandas as pd

from core.config import Config
from core.io_utils import read_parquet, write_parquet, write_json
from core.time_utils import hourly_range
from data_check import expected_raw_path


def preprocess(cfg: Config) -> pd.DataFrame:
    raw_path = expected_raw_path(cfg)
    df = read_parquet(raw_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index(df["ts"]).sort_index()
    rng = hourly_range(cfg.date_range.start, cfg.date_range.end)
    df = df.reindex(rng)
    if cfg.preprocess.get("drop_missing_candles", True):
        df = df.dropna()
    df = df[["open", "high", "low", "close", "volume", "quote_asset_volume"]]
    clean_path = cfg.paths.processed_dir / f"{cfg.symbol}_{cfg.timeframe}_clean.parquet"
    write_parquet(df.reset_index(), clean_path)
    meta = {"rows": len(df), "start": str(df.index.min()), "end": str(df.index.max())}
    write_json(meta, cfg.paths.processed_dir / f"{cfg.symbol}_{cfg.timeframe}_clean_meta.json")
    return df

from __future__ import annotations

"""Check for presence and coverage of local raw data."""

from pathlib import Path
import pandas as pd

from core.config import Config
from core.io_utils import read_parquet, write_parquet
from core.time_utils import hourly_range


def expected_raw_path(cfg: Config) -> Path:
    dr = cfg.date_range
    fname = f"{cfg.symbol}_{cfg.timeframe}_{dr.start}_{dr.end}.parquet"
    return cfg.paths.raw_dir / fname


def check_local_data(cfg: Config) -> bool:
    path = expected_raw_path(cfg)
    if not path.exists():
        return False
    df = read_parquet(path)
    if "ts" not in df.columns:
        # assume first column is timestamp in ms
        df["ts"] = pd.to_datetime(df.iloc[:, 0], unit="ms", utc=True)
    idx = pd.to_datetime(df["ts"], utc=True)
    idx = idx.dt.floor("H")
    df = df.set_index(idx).sort_index()
    rng = hourly_range(cfg.date_range.start, cfg.date_range.end)
    reindexed = df.reindex(rng)
    if reindexed.isnull().any().any():
        if cfg.data_check.get("keep_longest_contiguous", False):
            mask = reindexed["open"].notna()
            groups = mask.ne(mask.shift()).cumsum()
            sizes = reindexed.groupby(groups).size()
            best_group = sizes.loc[mask.groupby(groups).any()].idxmax()
            segment = reindexed[groups == best_group].dropna()
            write_parquet(segment.reset_index(), path)
            return True
        return False
    return True

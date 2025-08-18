from __future__ import annotations

"""Target variable creation."""

import numpy as np
import pandas as pd

from core.config import Config
from core.io_utils import write_parquet, write_json


def add_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["next_1h_log_return"] = np.log(df["close"].shift(-1) / df["close"])
    df = df.iloc[:-1]
    ml_path = cfg.paths.processed_dir / f"{cfg.symbol}_{cfg.timeframe}_ml.parquet"
    write_parquet(df.reset_index(), ml_path)
    columns_path = cfg.paths.processed_dir / f"{cfg.symbol}_{cfg.timeframe}_columns.txt"
    write_json({"columns": df.columns.tolist()}, columns_path)
    return df

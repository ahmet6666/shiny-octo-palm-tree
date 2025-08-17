"""Preprocess raw data and create features and targets."""
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
import pytz
import yaml
from sklearn.preprocessing import StandardScaler, RobustScaler

from utils.io import get_paths, read_parquet, write_parquet, write_json
from utils.logging_setup import setup_logging
from features import add_indicators

logger = logging.getLogger(__name__)


def build_dataset(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_index()
    if config["preprocess"].get("resample_to_strict_grid", True):
        rng = pd.date_range(df.index[0], df.index[-1], freq="1H", tz=pytz.UTC)
        df = df.reindex(rng)
    if config["preprocess"].get("drop_missing", True):
        df = df.dropna()

    df = df[["open", "high", "low", "close", "volume"]]
    df = add_indicators(df, config)
    df["log_return_t+1"] = np.log(df["close"].shift(-1) / df["close"])
    df = df.dropna()
    return df


def split_and_scale(df: pd.DataFrame, config: Dict) -> tuple[pd.DataFrame, Dict[str, float]]:
    n = len(df)
    train_end = int(n * config["preprocess"]["split"]["train_ratio"])
    val_end = train_end + int(n * config["preprocess"]["split"]["val_ratio"])
    df["split"] = "test"
    df.iloc[:train_end, df.columns.get_loc("split")] = "train"
    df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"

    feature_cols = [c for c in df.columns if c not in ["log_return_t+1", "split"]]
    scaler_info = {}
    if config["preprocess"]["scaling"]["enabled"]:
        method = config["preprocess"]["scaling"]["method"]
        scaler = StandardScaler() if method == "standard" else RobustScaler()
        train_df = df[df["split"] == "train"]
        scaler.fit(train_df[feature_cols])
        df[feature_cols] = scaler.transform(df[feature_cols])
        scaler_info = {
            "method": method,
            "mean": getattr(scaler, "mean_", []).tolist() if hasattr(scaler, "mean_") else [],
            "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else []
        }
    return df, scaler_info


def main(cfg_path: str):
    setup_logging()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = get_paths(config)
    df_raw = read_parquet(paths.raw_file)
    df = build_dataset(df_raw, config)
    df, scaler_info = split_and_scale(df, config)

    write_parquet(df, paths.processed_file)
    meta = {
        "columns": df.columns.tolist(),
        "scaler": scaler_info,
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
    }
    write_json(meta, paths.meta_file)
    logger.info("Saved processed data with %d rows", len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

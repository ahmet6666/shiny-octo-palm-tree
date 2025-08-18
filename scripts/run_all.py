from __future__ import annotations

"""Run full pipeline end-to-end."""

import argparse
import sys

from pathlib import Path

import pandas as pd

from pathlib import Path as _P; sys.path.append(str(_P(__file__).resolve().parents[1]/"src"))

from core.config import load_config
from data_check import check_local_data
from download import download_klines_binance
from preprocess import preprocess
from features import build_features
from target import add_target
from dataset import prepare_dataset
from train_lgbm import train_lgbm
from train_lstm import train_lstm
from core.logging_utils import setup_logging


def run_all(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    setup_logging(cfg.paths.logs_dir)
    if not check_local_data(cfg):
        download_klines_binance(cfg)
    df = preprocess(cfg)
    df = build_features(df, cfg)
    df = add_target(df, cfg)
    datasets = prepare_dataset(df, cfg)
    metrics = train_lgbm(datasets, cfg)
    if cfg.lstm_baseline.get("enabled", False):
        train_lstm(datasets, cfg)
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_all(Path(args.config))

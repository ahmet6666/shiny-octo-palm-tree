from __future__ import annotations

"""Quick smoke test for the pipeline (100 hours)."""

from pathlib import Path
import argparse
import sys

import pandas as pd
import numpy as np

from pathlib import Path as _P; sys.path.append(str(_P(__file__).resolve().parents[1]/"src"))
from core.config import load_config
from core.time_utils import parse_utc
from download import download_klines_binance
from preprocess import preprocess
from features import build_features
from target import add_target
from dataset import prepare_dataset
from train_lgbm import train_lgbm


def smoke_test(cfg_path: Path) -> bool:
    cfg = load_config(cfg_path)
    end = parse_utc(cfg.date_range.end)
    start = end - pd.Timedelta(hours=100)
    cfg.date_range.start = start.strftime("%Y-%m-%d %H:%M:%S")
    cfg.data_check["keep_longest_contiguous"] = False
    download_klines_binance(cfg)
    df = preprocess(cfg)
    df = build_features(df, cfg)
    df = add_target(df, cfg)
    datasets = prepare_dataset(df, cfg)
    cfg.train["n_trials"] = 1
    cfg.train["early_stopping_rounds"] = 10
    metrics = train_lgbm(datasets, cfg)
    ok = np.isfinite(metrics["r2"])
    print("Smoke test metrics:", metrics)
    if ok:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    success = smoke_test(Path(args.config))
    raise SystemExit(0 if success else 1)

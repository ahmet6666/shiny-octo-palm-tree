"""Evaluate trained model on test set."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import r2_score

from utils.io import get_paths, read_parquet, write_json
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def evaluate(config: dict) -> dict:
    paths = get_paths(config)
    df = read_parquet(paths.processed_file)
    fi_path = Path("./artifacts/reports/metrics.json")
    import pickle
    model_path = Path("./artifacts/models/lgbm_best.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train model first.")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    feature_cols = [c for c in df.columns if c not in ["log_return_t+1", "split"]]
    X_test = df[df["split"] == "test"][feature_cols]
    y_test = df[df["split"] == "test"]["log_return_t+1"]
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    metrics = {"test_r2": r2}
    write_json(metrics, fi_path)
    return metrics


def main(cfg_path: str):
    setup_logging()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = evaluate(config)
    logger.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

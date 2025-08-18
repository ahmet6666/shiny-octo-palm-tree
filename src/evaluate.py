from __future__ import annotations

"""Evaluation utilities."""

from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from core.config import Config
from core.io_utils import write_json


def evaluate_model(model, dataset: Tuple[pd.DataFrame, pd.Series], cfg: Config) -> Dict[str, float]:
    X_test, y_test = dataset
    pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred, squared=False),
    }
    write_json(metrics, cfg.paths.reports_dir / "evaluation.json")
    return metrics

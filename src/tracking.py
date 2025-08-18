from __future__ import annotations

"""MLflow tracking helpers."""

from contextlib import nullcontext

import mlflow

from core.config import Config


def mlflow_run(cfg: Config, run_name: str = "run"):
    if cfg.tracking.get("mlflow", {}).get("enabled", False):
        mlflow.set_tracking_uri(cfg.tracking["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg.tracking["mlflow"]["experiment_name"])
        return mlflow.start_run(run_name=run_name)
    return nullcontext()

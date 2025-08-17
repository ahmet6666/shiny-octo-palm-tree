"""Helpers for MLflow tracking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import mlflow


def start_run(experiment_name: str, run_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict[str, Any]):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: Dict[str, float], step: int | None = None):
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path):
    mlflow.log_artifact(str(path))

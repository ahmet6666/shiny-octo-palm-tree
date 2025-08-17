"""Train LightGBM model with Optuna HPO."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import r2_score

from utils.io import get_paths, read_parquet, write_json
from utils.logging_setup import setup_logging
from utils.mlflow_utils import start_run, log_params, log_metrics, log_artifact
from utils.eta_tracker import EtaTracker

logger = logging.getLogger(__name__)


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ["log_return_t+1", "split"]]
    X_train = df[df["split"] == "train"][feature_cols]
    y_train = df[df["split"] == "train"]["log_return_t+1"]
    X_val = df[df["split"] == "val"][feature_cols]
    y_val = df[df["split"] == "val"]["log_return_t+1"]
    X_test = df[df["split"] == "test"][feature_cols]
    y_test = df[df["split"] == "test"]["log_return_t+1"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, cfg_train: Dict, ui_cfg: Dict, progress_path: Path):
    params = {
        "objective": "regression",
        "metric": "r2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    eta = EtaTracker(start_batch=ui_cfg["eta_batch_start"], window=ui_cfg["eta_smoothing_window"])
    total_iters = cfg_train["num_boost_round"]
    last_time = time.time()

    def callback(env: lgb.CallbackEnv):
        nonlocal last_time
        now = time.time()
        duration = now - last_time
        last_time = now
        eta_sec = eta.update(env.iteration, duration, total_iters)
        progress = (env.iteration + 1) / total_iters
        write_json({"progress": progress, "eta_sec": eta_sec}, progress_path)

    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=cfg_train["num_boost_round"],
        early_stopping_rounds=cfg_train["early_stopping_rounds"],
        callbacks=[callback]
    )
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    trial.set_user_attr("model", model)
    return r2


def main(cfg_path: str):
    setup_logging()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = get_paths(config)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(paths.processed_file)

    progress_path = Path("./artifacts/reports/train_progress.json")
    cfg_train = config["training"]["lgbm"]

    ui_cfg = config["ui"]

    def opt_objective(trial: optuna.Trial):
        return objective(trial, X_train, y_train, X_val, y_val, cfg_train, ui_cfg, progress_path)

    study = optuna.create_study(direction="maximize")
    study.optimize(opt_objective, n_trials=cfg_train["optuna_trials"])
    best_model: lgb.Booster = study.best_trial.user_attrs["model"]

    preds = best_model.predict(X_test)
    test_r2 = r2_score(y_test, preds)

    model_path = Path("./artifacts/models/lgbm_best.pkl")
    with model_path.open("wb") as f:
        pickle.dump(best_model, f)

    # feature importance
    fi = pd.DataFrame({"feature": best_model.feature_name(), "importance": best_model.feature_importance()})
    fi_path = Path("./artifacts/reports/feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    if config["tracking"]["mlflow"]["enabled"]:
        mlcfg = config["tracking"]["mlflow"]
        with start_run(mlcfg["experiment_name"], "lgbm", mlcfg["tracking_uri"]):
            log_params(study.best_trial.params)
            log_metrics({"val_r2": study.best_value, "test_r2": test_r2})
            log_artifact(model_path)
            log_artifact(fi_path)

    write_json({"test_r2": test_r2}, Path("./artifacts/reports/metrics.json"))
    logger.info("Training complete. Test R2=%.4f", test_r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

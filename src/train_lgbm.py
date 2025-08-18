from __future__ import annotations

"""LightGBM training with Optuna."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import r2_score

from core.config import Config
from core.io_utils import write_json
from core.progress import BatchProgress


def train_lgbm(datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]], cfg: Config) -> Dict[str, float]:
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    progress = BatchProgress(cfg.train.get("n_trials", 10))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        }
        model = lgb.LGBMRegressor(random_state=cfg.train["random_state"], **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(cfg.train["early_stopping_rounds"], verbose=False)],
        )
        pred = model.predict(X_val)
        return r2_score(y_val, pred)

    def objective_wrapper(trial: optuna.Trial) -> float:
        score = objective(trial)
        progress.update()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective_wrapper,
        n_trials=cfg.train.get("n_trials", 10),
        timeout=cfg.train.get("timeout_seconds"),
    )

    best_params = study.best_trial.params
    model = lgb.LGBMRegressor(random_state=cfg.train["random_state"], **best_params)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    model_path = cfg.paths.artifacts_dir / "models" / "lgbm.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    metrics = {"r2": test_r2}
    write_json(metrics, cfg.paths.reports_dir / "metrics.json")
    pred_path = cfg.paths.reports_dir / "test_predictions.parquet"
    pd.DataFrame({"pred": test_pred, "true": y_test}).to_parquet(pred_path)
    return metrics

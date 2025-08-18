from __future__ import annotations

"""Dataset assembly and scaling."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.config import Config


def prepare_dataset(df: pd.DataFrame, cfg: Config) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    features = df.drop(columns=["next_1h_log_return"])
    target = df["next_1h_log_return"]
    n = len(df)
    train_end = int(n * cfg.split["train"])
    val_end = train_end + int(n * cfg.split["val"])

    X_train = features.iloc[:train_end]
    y_train = target.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    y_val = target.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]
    y_test = target.iloc[val_end:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    scaler_path = cfg.paths.artifacts_dir / "models" / "scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return {
        "train": (X_train_scaled, y_train),
        "val": (X_val_scaled, y_val),
        "test": (X_test_scaled, y_test),
    }

from __future__ import annotations

"""Optional LSTM baseline."""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

from core.config import Config
from core.progress import BatchProgress
from core.io_utils import write_json


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _seq_data(X: pd.DataFrame, y: pd.Series, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = X.values.astype(np.float32)
    tgt = y.values.astype(np.float32)
    xs, ys = [], []
    for i in range(len(arr) - seq_len):
        xs.append(arr[i : i + seq_len])
        ys.append(tgt[i + seq_len])
    return torch.tensor(xs), torch.tensor(ys)


def train_lstm(datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]], cfg: Config) -> Dict[str, float]:
    if not cfg.lstm_baseline.get("enabled", False):
        return {}

    params = cfg.lstm_baseline
    seq_len = params["seq_len"]
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_train_t, y_train_t = _seq_data(X_train, y_train, seq_len)
    X_val_t, y_val_t = _seq_data(X_val, y_val, seq_len)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=params["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(X_train.shape[1], params["hidden_size"], params["num_layers"], params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    best_r2 = -float("inf")
    progress = BatchProgress(params["epochs"])
    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                trues.append(yb.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        r2 = r2_score(trues, preds)
        progress.update()
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), cfg.paths.artifacts_dir / "models" / "lstm.pt")
    metrics = {"val_r2": float(best_r2)}
    write_json(metrics, cfg.paths.reports_dir / "lstm_metrics.json")
    return metrics

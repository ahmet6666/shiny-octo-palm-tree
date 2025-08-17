"""Train LSTM baseline."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

from utils.io import get_paths, read_parquet, write_json
from utils.logging_setup import setup_logging
from utils.mlflow_utils import start_run, log_params, log_metrics, log_artifact
from utils.eta_tracker import EtaTracker

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], seq_len: int):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df["log_return_t+1"].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]
        return x, y


class LSTMReg(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


def train_model(model, loaders, cfg, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best_val = float("inf")
    patience = cfg["early_stopping_patience"]
    eta_tracker = EtaTracker(start_batch=cfg["eta_batch_start"], window=cfg["eta_smoothing_window"])
    progress_path = Path("./artifacts/reports/train_progress.json")

    for epoch in range(cfg["max_epochs"]):
        model.train()
        start = time.time()
        for i, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            duration = time.time() - start
            eta = eta_tracker.update(i, duration, len(loaders["train"]))
            progress = (epoch + i / len(loaders["train"])) / cfg["max_epochs"]
            write_json({"progress": progress, "eta_sec": eta}, progress_path)
            start = time.time()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_losses.append(criterion(preds, y).item())
            val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), "./artifacts/models/lstm_best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    return model


def main(cfg_path: str):
    setup_logging()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = get_paths(config)
    df = read_parquet(paths.processed_file)
    cfg_train = config["training"]["lstm"]
    seed_everything(config["project"]["seed"])
    feature_cols = [c for c in df.columns if c not in ["log_return_t+1", "split"]]

    seq_len = cfg_train["seq_len"]
    datasets = {}
    loaders = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        datasets[split] = SeqDataset(split_df, feature_cols, seq_len)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=cfg_train["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg_train["num_workers"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMReg(len(feature_cols), cfg_train["hidden_size"], cfg_train["num_layers"], cfg_train["dropout"])
    model.to(device)

    train_model(model, loaders, {**cfg_train, **config["ui"]}, device)

    # Evaluation
    model.load_state_dict(torch.load("./artifacts/models/lstm_best.pt", map_location=device))
    model.eval()
    preds_list, y_list = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            preds_list.append(preds)
            y_list.append(y.numpy())
    preds = np.concatenate(preds_list)
    y_true = np.concatenate(y_list)
    test_r2 = r2_score(y_true, preds)

    write_json({"test_r2_lstm": test_r2}, Path("./artifacts/reports/metrics_lstm.json"))
    if config["tracking"]["mlflow"]["enabled"]:
        mlcfg = config["tracking"]["mlflow"]
        with start_run(mlcfg["experiment_name"], "lstm", mlcfg["tracking_uri"]):
            log_params(cfg_train)
            log_metrics({"test_r2": test_r2})
            log_artifact(Path("./artifacts/models/lstm_best.pt"))
    logger.info("LSTM test R2=%.4f", test_r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

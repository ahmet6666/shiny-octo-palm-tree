from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

from core.config import load_config
from data_check import check_local_data, expected_raw_path
from download import download_klines_binance
from preprocess import preprocess
from features import build_features
from target import add_target
from dataset import prepare_dataset
from train_lgbm import train_lgbm


CFG_PATH = Path("configs/config.yaml")


def main():
    cfg = load_config(CFG_PATH)
    st.title("BTC/USDT 1h Regression")

    if st.button("Check & Download Data"):
        if not check_local_data(cfg):
            download_klines_binance(cfg)
            st.write("Downloaded")
        else:
            st.write("Data exists")

    if st.button("Preprocess + Features + Target"):
        df = preprocess(cfg)
        df = build_features(df, cfg)
        df = add_target(df, cfg)
        st.write(f"Prepared {len(df)} rows")

    if st.button("Train LightGBM"):
        ml_path = cfg.paths.processed_dir / f"{cfg.symbol}_{cfg.timeframe}_ml.parquet"
        df = pd.read_parquet(ml_path)
        datasets = prepare_dataset(df, cfg)
        metrics = train_lgbm(datasets, cfg)
        st.write(metrics)


if __name__ == "__main__":
    main()

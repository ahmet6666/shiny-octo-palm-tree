"""Streamlit dashboard for end-to-end pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import yaml

from data_check import check_local_data
from download_binance import main as download_main
from preprocess import main as preprocess_main
from train_lgbm import main as train_lgbm_main
from train_lstm import main as train_lstm_main
from evaluate import main as evaluate_main

CFG_PATH = "configs/config.yaml"


def load_config() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def show_progress(refresh: int):
    progress_file = Path("./artifacts/reports/train_progress.json")
    progress_bar = st.progress(0.0)
    eta_placeholder = st.empty()
    while True:
        if progress_file.exists():
            try:
                with progress_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                prog = data.get("progress", 0.0)
                progress_bar.progress(min(1.0, prog))
                eta = data.get("eta_sec")
                if eta is not None:
                    eta_placeholder.write(f"ETA: {eta/60:.1f} dk")
                if prog >= 1.0:
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(refresh)


def main():
    st.title("BTC 1H Regresyon Pipeline")
    config = load_config()
    st.subheader("Konfigürasyon")
    st.json(config)

    if st.button("Veriyi Kontrol Et"):
        res = check_local_data(config)
        st.write(res)

    if st.button("İndir (Gerekirse)"):
        with st.spinner("İndiriliyor..."):
            download_main(CFG_PATH)
        st.success("Tamamlandı")

    if st.button("Ön İşleme"):
        with st.spinner("Ön işleniyor..."):
            preprocess_main(CFG_PATH)
        st.success("Tamamlandı")

    train_choice = st.selectbox("Eğit", ["LightGBM", "LSTM", "Both"], index=0)
    if st.button("Eğit"):
        if train_choice in ("LightGBM", "Both"):
            with st.spinner("LightGBM eğitiliyor..."):
                import threading

                thread = threading.Thread(target=train_lgbm_main, args=(CFG_PATH,))
                thread.start()
                show_progress(config["ui"]["refresh_every_sec"])
                thread.join()
        if train_choice in ("LSTM", "Both"):
            with st.spinner("LSTM eğitiliyor..."):
                import threading

                thread = threading.Thread(target=train_lstm_main, args=(CFG_PATH,))
                thread.start()
                show_progress(config["ui"]["refresh_every_sec"])
                thread.join()
        st.success("Eğitim tamamlandı")

    if st.button("Değerlendir"):
        with st.spinner("Değerlendiriliyor..."):
            evaluate_main(CFG_PATH)
        st.success("Tamamlandı")


if __name__ == "__main__":
    main()

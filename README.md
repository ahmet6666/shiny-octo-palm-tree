# BTC/USDT 1h Regression Pipeline

This repository provides a complete workflow for predicting the next hour logarithmic return of BTC/USDT using 1‑hour candles from Binance Spot market. It covers data acquisition, preprocessing, feature engineering, model training with LightGBM, optional LSTM baseline, evaluation and a small Streamlit dashboard.

## Quick Start

```bash
python scripts/make_dirs.py --config configs/config.yaml
python scripts/run_all.py --config configs/config.yaml
```

Launch the Streamlit UI:

```bash
streamlit run src/ui_app.py
```

Run the smoke test (uses only the last 100 hours):

```bash
python scripts/smoke_test.py --config configs/config.yaml
```

## Data Policy
- All times are in **UTC** and aligned to a strict hourly grid.
- Missing candles are dropped; optionally the longest contiguous segment is kept.
- Raw data and processed datasets are stored under `veriler/`.

## Features & Target
- Technical indicators: RSI, MACD, EMA(12/26/50/200), Bollinger Bands, Stochastic, ATR.
- Additional features: log price, 1‑hour log return.
- Target: `next_1h_log_return = ln(close[t+1]/close[t])`.

## MLflow
Experiment tracking is enabled by default and writes runs to the local `mlruns` directory.

## Troubleshooting
- Binance API rate limits: retry or provide API keys via environment variables if needed.
- Timezone errors: ensure system clock uses UTC.
- Missing data: the `data_check` step validates coverage before training.

## License
MIT

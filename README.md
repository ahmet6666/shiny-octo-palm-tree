# BTC 1H Regression

This project predicts the next hour log return of BTC/USDT using historical 1-hour candlestick data. It covers the full pipeline from data download to model training, evaluation, and visualization via Streamlit.

## Features
- Download 2020-2022 Binance spot klines using `python-binance`.
- Technical indicators via `ta`.
- LightGBM with Optuna hyper-parameter optimization.
- Optional LSTM baseline with PyTorch.
- MLflow experiment tracking.
- Streamlit dashboard in Turkish.

## Installation
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```
python scripts/data_check.py --config configs/config.yaml
python scripts/download_binance.py --config configs/config.yaml
python scripts/preprocess.py --config configs/config.yaml
python scripts/train_lgbm.py --config configs/config.yaml
python scripts/evaluate.py --config configs/config.yaml
```

Dashboard:
```
streamlit run scripts/ui_app.py
```

## Structure
```
configs/
  config.yaml
scripts/
  ...
veriler/
  raw/
  processed/
artifacts/
  models/
  logs/
  reports/
```

## MLflow
Set tracking URI in `config.yaml`. Runs are stored under `artifacts/mlruns` by default.

## Issues
- Binance API limits may cause slow downloads.
- Ensure system time is in UTC.
- Missing candles will be dropped.

## License
MIT

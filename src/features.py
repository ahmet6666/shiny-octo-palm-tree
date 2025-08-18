from __future__ import annotations

"""Feature engineering using technical indicators."""

import pandas as pd
import numpy as np
import ta

from core.config import Config
from core.io_utils import write_parquet


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    # basic transforms
    df["log_close"] = (df["close"].astype(float)).apply(lambda x: np.log(x))
    df["log_return_1h"] = df["log_close"].diff()

    # RSI
    rsi_period = cfg.technical_indicators["rsi"]["period"]
    df["rsi"] = ta.momentum.RSIIndicator(df["close"].astype(float), rsi_period).rsi()

    # MACD
    macd_cfg = cfg.technical_indicators["macd"]
    macd = ta.trend.MACD(
        df["close"].astype(float),
        macd_cfg["fast"],
        macd_cfg["slow"],
        macd_cfg["signal"],
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # EMA
    for span in cfg.technical_indicators["ema"]:
        df[f"ema_{span}"] = ta.trend.EMAIndicator(df["close"].astype(float), span).ema_indicator()

    # Bollinger Bands
    bb_cfg = cfg.technical_indicators["bb"]
    bb = ta.volatility.BollingerBands(df["close"].astype(float), bb_cfg["period"], bb_cfg["std"])
    df["bb_h"] = bb.bollinger_hband()
    df["bb_l"] = bb.bollinger_lband()

    # Stochastic
    stoch_cfg = cfg.technical_indicators["stoch"]
    stoch = ta.momentum.StochasticOscillator(
        df["high"].astype(float),
        df["low"].astype(float),
        df["close"].astype(float),
        stoch_cfg["k"],
        stoch_cfg["d"],
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ATR
    atr_period = cfg.technical_indicators["atr"]["period"]
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"].astype(float), df["low"].astype(float), df["close"].astype(float), atr_period
    ).average_true_range()

    df = df.dropna()
    feat_path = cfg.paths.processed_dir / f"{cfg.symbol}_{cfg.timeframe}_features.parquet"
    write_parquet(df.reset_index(), feat_path)
    return df

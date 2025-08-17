"""Feature engineering with technical indicators."""
from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["preprocess"]["features"]
    if cfg.get("rsi"):
        period = cfg["rsi"]["period"]
        df[f"rsi_{period}"] = RSIIndicator(close=df["close"], window=period).rsi()
    if cfg.get("macd"):
        m = cfg["macd"]
        macd = MACD(close=df["close"], window_slow=m["slow"], window_fast=m["fast"], window_sign=m["signal"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
    if cfg.get("ema"):
        for period in cfg["ema"]:
            df[f"ema_{period}"] = EMAIndicator(close=df["close"], window=period).ema_indicator()
    if cfg.get("bb"):
        bb = cfg["bb"]
        bands = BollingerBands(close=df["close"], window=bb["period"], window_dev=bb["std"])
        df[f"bb_mid_{bb['period']}_{bb['std']}"] = bands.bollinger_mavg()
        df[f"bb_upper_{bb['period']}_{bb['std']}"] = bands.bollinger_hband()
        df[f"bb_lower_{bb['period']}_{bb['std']}"] = bands.bollinger_lband()
    if cfg.get("stoch"):
        st = cfg["stoch"]
        so = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=st["k"], smooth_window=st["d"])
        df[f"stoch_k_{st['k']}_{st['d']}"] = so.stoch()
        df[f"stoch_d_{st['k']}_{st['d']}"] = so.stoch_signal()
    if cfg.get("atr"):
        period = cfg["atr"]["period"]
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period)
        df[f"atr_{period}"] = atr.average_true_range()

    df = df.dropna().copy()
    return df

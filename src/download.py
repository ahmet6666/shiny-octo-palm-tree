from __future__ import annotations

"""Download kline data from Binance."""

import time
from pathlib import Path
from typing import List

import pandas as pd
from binance.client import Client

from core.config import Config
from core.io_utils import write_parquet
from core.time_utils import parse_utc


def download_klines_binance(cfg: Config) -> pd.DataFrame:
    start = parse_utc(cfg.date_range.start)
    end = parse_utc(cfg.date_range.end)
    client = Client()
    klines: List[List] = []
    limit = 1000
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while start_ms < end_ms:
        batch = client.get_klines(
            symbol=cfg.symbol,
            interval=Client.KLINE_INTERVAL_1HOUR,
            startTime=start_ms,
            endTime=end_ms,
            limit=limit,
        )
        if not batch:
            break
        klines.extend(batch)
        start_ms = batch[-1][0] + 3600_000
        time.sleep(0.5)  # rate limit
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "n_trades",
        "taker_base_vol",
        "taker_quote_vol",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    write_parquet(df, Path(cfg.paths.raw_dir) / f"{cfg.symbol}_{cfg.timeframe}_{cfg.date_range.start}_{cfg.date_range.end}.parquet")
    return df

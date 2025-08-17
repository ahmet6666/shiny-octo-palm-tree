"""Download historical klines from Binance."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from time import sleep

import pandas as pd
import pytz
import yaml
from binance.client import Client
from binance.exceptions import BinanceAPIException

from utils.io import get_paths, ensure_dirs, write_parquet
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def fetch_klines(client: Client, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    limit = 1000
    start_ts = int(pd.Timestamp(start, tz=pytz.UTC).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end, tz=pytz.UTC).timestamp() * 1000)
    all_klines = []
    while start_ts < end_ts:
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_ts, endTime=end_ts, limit=limit)
        except BinanceAPIException as e:
            logger.error("Binance API error: %s", e)
            sleep(1)
            continue
        if not klines:
            break
        all_klines.extend(klines)
        start_ts = klines[-1][0] + 1
        sleep(0.5)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_klines, columns=cols)
    df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})
    return df


def main(cfg_path: str):
    setup_logging()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = get_paths(config)
    ensure_dirs(paths.raw_dir)

    data_cfg = config["data"]
    client = Client()
    df = fetch_klines(client, data_cfg["symbol"], Client.KLINE_INTERVAL_1HOUR, data_cfg["start_date"], data_cfg["end_date"])
    write_parquet(df, paths.raw_file)
    logger.info("Downloaded %d rows", len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

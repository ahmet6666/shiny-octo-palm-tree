"""Fetch 1-hour candlestick data from Binance and train a simple ML model.

This script downloads the last 240 hours of BTC/USDT price data from the
Binance public REST API.  It then prepares a dataset for predicting the
next hour's closing price using a very small linear regression
implementation written with only Python's standard library.  The code is
heavily commented to explain each step.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants describing the data we fetch
# ---------------------------------------------------------------------------
# Try the global Binance endpoint first.  Some regions (such as the US)
# cannot access it and receive an HTTP 451 error.  For those cases we fall
# back to the Binance.US endpoint which exposes the same public data.
BINANCE_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
]
SYMBOL = "BTCUSDT"  # Bitcoin priced in USDT
INTERVAL = "1h"     # 1-hour candlesticks
LIMIT = 240         # 240 hours = 10 days


# ---------------------------------------------------------------------------
# Data representation
# ---------------------------------------------------------------------------
@dataclass
class Candle:
    """Represents a single candlestick from Binance."""

    open_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def from_api_row(row: List[str]) -> "Candle":
        """Convert a raw API row into a :class:`Candle`.

        Parameters
        ----------
        row: list[str]
            A single entry from the Binance `/klines` endpoint.
        """

        open_time = dt.datetime.fromtimestamp(row[0] / 1000)
        return Candle(
            open_time=open_time,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def fetch_candles() -> List[Candle]:
    """Fetch the last ``LIMIT`` candles from Binance.

    The function attempts multiple endpoints to handle regional
    restrictions.  If the global Binance API returns an HTTP 451 (restricted
    location), we retry using the Binance.US endpoint.
    """

    params = f"?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
    last_error: Exception | None = None
    for base in BINANCE_URLS:
        try:
            with urllib.request.urlopen(base + params, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return [Candle.from_api_row(row) for row in raw]
        except urllib.error.HTTPError as e:
            last_error = e
            continue
    raise RuntimeError("Failed to fetch candles from any Binance endpoint") from last_error


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(candles: List[Candle]) -> Tuple[List[List[float]], List[float]]:
    """Prepare feature and target arrays for training.

    We use the current open, high, low, close, and volume to predict the
    next hour's closing price.  ``y`` therefore contains the shifted close
    price.
    """

    features: List[List[float]] = []
    target: List[float] = []
    for i in range(len(candles) - 1):
        c = candles[i]
        features.append([c.open, c.high, c.low, c.close, c.volume])
        target.append(candles[i + 1].close)
    return features, target


# ---------------------------------------------------------------------------
# Linear regression using gradient descent
# ---------------------------------------------------------------------------
def train_linear_regression(
    X: List[List[float]],
    y: List[float],
    epochs: int = 10_000,
    lr: float = 1e-12,
) -> Tuple[List[float], float]:
    """Train a linear regression model using batch gradient descent.

    Parameters
    ----------
    X: list[list[float]]
        Feature matrix where each inner list represents a sample.
    y: list[float]
        Target values (next closing price).
    epochs: int
        Number of gradient descent iterations.
    lr: float
        Learning rate controlling the update size.
    """

    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    m = len(X)

    for _ in range(epochs):
        # Predictions for each sample
        preds = [sum(w * x for w, x in zip(weights, row)) + bias for row in X]
        # Errors between prediction and true value
        errors = [p - t for p, t in zip(preds, y)]
        # Gradient for each weight
        grad_w = [sum(err * row[i] for err, row in zip(errors, X)) / m for i in range(n_features)]
        grad_b = sum(errors) / m
        # Parameter update
        weights = [w - lr * gw for w, gw in zip(weights, grad_w)]
        bias -= lr * grad_b
    return weights, bias


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """Compute the mean absolute error between predictions and true values."""

    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


# ---------------------------------------------------------------------------
# Main entry point tying everything together
# ---------------------------------------------------------------------------
def main() -> None:
    candles = fetch_candles()
    X, y = prepare_dataset(candles)
    weights, bias = train_linear_regression(X, y)

    # Evaluate on the last 20% of samples
    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]
    preds = [sum(w * x for w, x in zip(weights, row)) + bias for row in X_test]
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean absolute error on test set: {mae:.2f} USDT")
    print("Learned weights:", weights)
    print("Bias:", bias)


if __name__ == "__main__":
    main()

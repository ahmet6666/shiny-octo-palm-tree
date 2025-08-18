from __future__ import annotations

"""Timezone helpers."""

from datetime import datetime
import pandas as pd
import pytz


UTC = pytz.UTC


def parse_utc(ts: str | datetime) -> pd.Timestamp:
    """Parse a datetime string or object and return a UTC-aware Timestamp."""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.tz_localize(UTC)
        return ts.tz_convert(UTC)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return pd.Timestamp(ts, tz=UTC)
        return pd.Timestamp(ts).tz_convert(UTC)
    return pd.to_datetime(ts, utc=True)


def hourly_range(start: str | datetime, end: str | datetime) -> pd.DatetimeIndex:
    """Return an inclusive hourly range [start, end]."""
    s = parse_utc(start)
    e = parse_utc(end)
    return pd.date_range(s, e, freq="1H", tz=UTC)

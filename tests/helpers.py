"""Small deterministic data builders shared by tests."""

from datetime import UTC, datetime

import pandas as pd


def klines(count: int = 300, start: datetime = datetime(2026, 1, 1, tzinfo=UTC)) -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="1min")
    close = pd.Series(range(count), index=index, dtype="float64") * 0.01 + 100
    return pd.DataFrame(
        {
            "o": close - 0.005,
            "h": close + 0.02,
            "l": close - 0.02,
            "c": close,
            "v": 1000.0,
        },
        index=index,
    )


def flow(events: list[tuple[datetime, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cls": [5] * len(events),
            "side": [event[1] for event in events],
            "price": [100.0] * len(events),
            "vol": [event[2] for event in events],
        },
        index=pd.DatetimeIndex([event[0] for event in events], name="t", tz="UTC"),
    )
    return frame.astype({"cls": "int64", "side": "str", "price": "float64", "vol": "float64"})

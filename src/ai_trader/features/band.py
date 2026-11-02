"""Causal EMA9/BB20/MA60/Ichimoku band votes."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_BARS = 60


def add_band_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Add band indicators and a four-vote trend using only each row and its past.

    Input bars must be completed and indexed by their close time. The Ichimoku
    vote compares close with the contemporaneous cloud; no displaced future span
    is read, which avoids introducing lookahead.
    """

    frame = bars.copy()
    close = frame["c"]
    high = frame["h"]
    low = frame["l"]
    frame["ema9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    frame["bb_mid"] = close.rolling(20, min_periods=20).mean()
    frame["ma60"] = close.rolling(60, min_periods=60).mean()
    tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2
    kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2
    span_b = (high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2
    frame["span_a"] = (tenkan + kijun) / 2
    frame["span_b"] = span_b

    votes = pd.DataFrame(index=frame.index)
    votes["ema9"] = _vote(close, frame["ema9"])
    votes["bb20"] = _vote(close, frame["bb_mid"])
    votes["ma60"] = _vote(close, frame["ma60"])
    cloud_top = frame[["span_a", "span_b"]].max(axis=1)
    cloud_bottom = frame[["span_a", "span_b"]].min(axis=1)
    votes["ichimoku"] = np.select(
        [close > cloud_top, close < cloud_bottom], [1, -1], default=0
    )
    valid = frame[["ema9", "bb_mid", "ma60", "span_a", "span_b"]].notna().all(axis=1)
    frame["band_score"] = votes.sum(axis=1).where(valid)
    frame["trend"] = np.select(
        [frame["band_score"] >= 3, frame["band_score"] <= -3],
        ["bull", "bear"],
        default="neutral",
    )
    frame.loc[~valid, "trend"] = "neutral"
    return frame


def _vote(close: pd.Series, indicator: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([close > indicator, close < indicator], [1, -1], default=0),
        index=close.index,
        dtype="int64",
    )

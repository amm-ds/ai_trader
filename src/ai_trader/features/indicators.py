"""Causal indicators shared by the baseline strategies and regime router."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ai_trader.features.band import add_band_features


def completed_history(bars: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Slice a sorted frame at its right-closed timestamp without a full scan."""

    end = int(bars.index.searchsorted(pd.Timestamp(as_of), side="right"))
    return bars.iloc[:end]


def add_strategy_features(bars: pd.DataFrame, percentile_window: int = 1920) -> pd.DataFrame:
    """Add completed-bar ATR, RSI, ROC, Bollinger width, and causal percentiles."""

    frame = add_band_features(bars)
    close = frame["c"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["h"] - frame["l"],
            (frame["h"] - previous_close).abs(),
            (frame["l"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr14"] = true_range.rolling(14, min_periods=14).mean()

    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    frame["rsi3"] = (100 - 100 / (1 + relative_strength)).fillna(50.0)
    frame.loc[(average_loss == 0) & (average_gain > 0), "rsi3"] = 100.0
    frame.loc[(average_loss == 0) & (average_gain == 0), "rsi3"] = 50.0

    rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
    frame["bb_width"] = (4 * rolling_std / frame["bb_mid"].abs()).replace(
        [np.inf, -np.inf], np.nan
    )
    log_returns = np.log(close / previous_close).replace([np.inf, -np.inf], np.nan)
    frame["realized_vol"] = log_returns.rolling(20, min_periods=20).std(ddof=0)
    minimum = min(100, percentile_window)
    for column in ("bb_width", "realized_vol"):
        frame[f"{column}_pct"] = frame[column].rolling(
            percentile_window, min_periods=minimum
        ).rank(pct=True)
    frame["roc_1"] = close.pct_change(1, fill_method=None)
    frame["volume_z"] = (frame["v"] - frame["v"].rolling(20, min_periods=20).mean()) / frame[
        "v"
    ].rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    return frame

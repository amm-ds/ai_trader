"""Deterministic, HMM-free market regime classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from ai_trader.features.indicators import completed_history

RegimeLabel = Literal["TREND", "RANGE", "VOLATILE", "TRANSITION"]


@dataclass(frozen=True, slots=True)
class RegimeState:
    """The regime and the causal evidence used to route a strategy review."""

    label: RegimeLabel
    direction: Literal["bull", "bear", "neutral"]
    aligned_timeframes: int
    bb_width_percentile: float
    realized_vol_percentile: float


class RegimeClassifier:
    """Classify volatility first, then band alignment, without fitted state."""

    def __init__(self, volatility_percentile: float = 0.95) -> None:
        if not 0.5 < volatility_percentile < 1.0:
            raise ValueError("volatility percentile must be in (0.5, 1.0)")
        self.volatility_percentile = volatility_percentile

    def evaluate(
        self, features: Mapping[str, pd.DataFrame], as_of: datetime
    ) -> RegimeState:
        """Return a regime using only completed rows at or before `as_of`."""

        trends: list[str] = []
        for timeframe in ("5m", "15m", "1h", "4h"):
            history = completed_history(features[timeframe], as_of)
            if history.empty:
                return RegimeState("TRANSITION", "neutral", 0, 0.0, 0.0)
            trends.append(str(history.iloc[-1]["trend"]))
        bull = trends.count("bull")
        bear = trends.count("bear")
        aligned = max(bull, bear)
        direction: Literal["bull", "bear", "neutral"] = "neutral"
        if bull > bear:
            direction = "bull"
        elif bear > bull:
            direction = "bear"
        execution = completed_history(features["15m"], as_of)
        row = execution.iloc[-1]
        bb_percentile = _finite(row.get("bb_width_pct"))
        vol_percentile = _finite(row.get("realized_vol_pct"))
        if max(bb_percentile, vol_percentile) >= self.volatility_percentile:
            label: RegimeLabel = "VOLATILE"
        elif aligned == 4:
            label = "TREND"
        elif aligned <= 1:
            label = "RANGE"
        else:
            label = "TRANSITION"
        return RegimeState(label, direction, aligned, bb_percentile, vol_percentile)


def _finite(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        result = float(value)
    except ValueError:
        return 0.0
    return result if pd.notna(result) else 0.0

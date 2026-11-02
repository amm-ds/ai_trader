"""Fixed chandelier ratchets and structural exits for trend-routed entries."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.indicators import completed_history
from ai_trader.signal.band_rejection import Side

CHANDELIER_ATR_MULTIPLE = 3.0
TREND_STRATEGIES = ("band_rejection", "momentum", "spring_reversal", "flow_continuation")


def trend_target(
    features: Mapping[str, pd.DataFrame], side: Side, as_of: datetime
) -> float | None:
    """Use the prior 5m swing, falling back to the completed 1h Bollinger band."""

    execution = completed_history(features["5m"], as_of)
    if len(execution) < 21:
        return None
    prior = execution.iloc[-21:-1]
    swing = float(prior["h"].max() if side == "long" else prior["l"].min())
    close = float(execution.iloc[-1]["c"])
    if pd.notna(swing) and ((side == "long" and swing > close) or (side == "short" and swing < close)):
        return swing
    higher = completed_history(features["1h"], as_of)
    if higher.empty:
        return None
    row = higher.iloc[-1]
    middle = float(row.get("bb_mid", float("nan")))
    width = float(row.get("bb_width", float("nan")))
    if not pd.notna(middle) or not pd.notna(width):
        return None
    band = middle * (1 + width / 2) if side == "long" else middle * (1 - width / 2)
    return band if (side == "long" and band > close) or (side == "short" and band < close) else None


def profit_ratchet_armed(
    side: Side,
    entry_price: float,
    initial_stop: float,
    highest: float,
    lowest: float,
) -> bool:
    """Arm profit ratchets only after price travels one initial price-R."""

    risk = abs(entry_price - initial_stop)
    return highest >= entry_price + risk if side == "long" else lowest <= entry_price - risk


def chandelier_stop(
    side: Side,
    entry_price: float,
    current_stop: float,
    highest: float,
    lowest: float,
    atr: float,
    armed: bool,
) -> float:
    """Ratchet to breakeven and a 3×ATR chandelier, never widening the stop."""

    if not armed:
        return current_stop
    candidates = [current_stop, entry_price]
    if pd.notna(atr) and atr > 0:
        chandelier = (
            highest - CHANDELIER_ATR_MULTIPLE * atr
            if side == "long"
            else lowest + CHANDELIER_ATR_MULTIPLE * atr
        )
        candidates.append(chandelier)
    return max(candidates) if side == "long" else min(candidates)


def structural_break(side: Side, execution: pd.Series, flow: FlowWindow) -> bool:
    """Require an EMA9 close-through and contemporaneous flow agreement."""

    close = float(execution["c"])
    ema9 = float(execution["ema9"])
    if not pd.notna(close) or not pd.notna(ema9):
        return False
    if side == "long":
        return close < ema9 and (flow.events == 0 or flow.delta < 0)
    return close > ema9 and (flow.events == 0 or flow.delta > 0)

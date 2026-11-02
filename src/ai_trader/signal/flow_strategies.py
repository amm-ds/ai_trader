"""Flow-confirmed spring reversal and regime continuation strategies."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.indicators import completed_history
from ai_trader.models.regime import RegimeClassifier, RegimeState
from ai_trader.signal.band_rejection import Side, TradeSignal
from ai_trader.signal.exits import trend_target


class FlowEnsembleStrategy:
    """Use bands only for regime/structure; flow confirms, never triggers alone."""

    def __init__(self, absorption_log_odds: float = 2.0, cvd_slope_quantile: float = 0.75) -> None:
        self.absorption_log_odds = absorption_log_odds
        self.cvd_slope_quantile = cvd_slope_quantile
        self.regime = RegimeClassifier()
        self.last_regime: RegimeState | None = None
        self.last_strategy = "none"
        self.last_reason = "not-evaluated"

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        state = self.regime.evaluate(features, as_of)
        self.last_regime = state
        if state.label == "VOLATILE":
            self.last_strategy = "none"
            self.last_reason = "regime-volatile-stand-down"
            return None
        execution = completed_history(features["15m"], as_of)
        flow_features = completed_history(features["flow"], as_of)
        if len(execution) < 2 or len(flow_features) < 2:
            return None
        if state.label in ("RANGE", "TRANSITION"):
            self.last_strategy = "spring_reversal"
            signal = self._spring(symbol, features, execution, flow_features)
        elif state.label == "TREND":
            self.last_strategy = "flow_continuation"
            signal = self._continuation(symbol, features, execution, flow_features, state.direction)
        else:
            return None
        self.last_reason = "signal" if signal is not None else "no-flow-structure-trigger"
        return signal

    def _spring(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        execution: pd.DataFrame,
        flow_features: pd.DataFrame,
    ) -> TradeSignal | None:
        row = execution.iloc[-1]
        previous = execution.iloc[-2]
        higher = completed_history(features["1h"], execution.index[-1])
        if higher.empty:
            return None
        htf = higher.iloc[-1]
        close = float(row["c"])
        low = float(row["l"])
        high = float(row["h"])
        trend = str(row["trend"])
        prior_trend = str(previous["trend"])
        log_odds = float(flow_features.iloc[-1]["absorption_log_odds_2h"])
        log_odds_6h = float(flow_features.iloc[-1]["absorption_log_odds_6h"])
        configurations: list[tuple[Side, str, float]] = [("long", "bull", low), ("short", "bear", high)]
        for side, expected, extreme in configurations:
            if trend != expected or prior_trend == expected:
                continue
            demand = any(
                extreme <= float(htf[column]) and close > float(htf[column])
                for column in ("bb_mid", "ma60")
            ) if side == "long" else any(
                extreme >= float(htf[column]) and close < float(htf[column])
                for column in ("bb_mid", "ma60")
            )
            absorption = (
                log_odds >= self.absorption_log_odds and log_odds_6h > 0
                if side == "long"
                else log_odds <= -self.absorption_log_odds and log_odds_6h < 0
            )
            if demand and absorption:
                return self._signal(symbol, side, execution, "spring_reversal", log_odds)
        return None

    def _continuation(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        execution: pd.DataFrame,
        flow_features: pd.DataFrame,
        direction: str,
    ) -> TradeSignal | None:
        if direction not in ("bull", "bear"):
            return None
        row = execution.iloc[-1]
        previous = execution.iloc[-2]
        current_flow = flow_features.iloc[-1]
        previous_flow = flow_features.iloc[-2]
        close = float(row["c"])
        ema9 = float(row["ema9"])
        side: Side = "long" if direction == "bull" else "short"
        if side == "long":
            pullback = float(previous["l"]) <= max(float(previous["ema9"]), float(previous["bb_mid"]))
            resume = pullback and close > ema9 and close > float(row["o"])
            delta_expands = float(current_flow["delta_15m"]) > float(previous_flow["delta_15m"]) and float(current_flow["delta_15m"]) > 0
            cvd_agrees = float(current_flow["cvd_change_15m"]) > 0
        else:
            pullback = float(previous["h"]) >= min(float(previous["ema9"]), float(previous["bb_mid"]))
            resume = pullback and close < ema9 and close < float(row["o"])
            delta_expands = float(current_flow["delta_15m"]) < float(previous_flow["delta_15m"]) and float(current_flow["delta_15m"]) < 0
            cvd_agrees = float(current_flow["cvd_change_15m"]) < 0
        strong = float(current_flow["cvd_slope_pct"]) >= self.cvd_slope_quantile
        if not (resume and delta_expands and cvd_agrees and strong):
            return None
        return self._signal(symbol, side, execution, "flow_continuation", float(current_flow["absorption_log_odds_2h"]))

    @staticmethod
    def _signal(
        symbol: str,
        side: Side,
        execution: pd.DataFrame,
        thesis: str,
        log_odds: float,
    ) -> TradeSignal:
        row = execution.iloc[-1]
        close = float(row["c"])
        atr = float(row["atr14"])
        buffer_ = max(0.1 * atr, 0.001 * close)
        stop = float(row["l"]) - buffer_ if side == "long" else float(row["h"]) + buffer_
        return TradeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_price=stop,
            target_price=trend_target({"5m": execution, "1h": execution}, side, execution.index[-1]),
            thesis=f"{thesis}-{side}",
            correlation_group="crypto-beta",
            trend="bull" if side == "long" else "bear",
            cascade_confirmed=True,
            written_trigger="15m structure trigger with causal flow confirmation",
            conditions=(f"flow-log-odds:{log_odds:.6f}", "15m-structure-confirmed"),
        )

"""Regime-gated routing across the P1 baseline strategies."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.models.mean_reversion import MeanReversionStrategy
from ai_trader.models.momentum import MomentumStrategy
from ai_trader.models.regime import RegimeClassifier, RegimeState
from ai_trader.signal.band_rejection import BandRejectionStrategy, TradeSignal


class EnsembleStrategy:
    """Route trend/range signals and stand down in volatile/transition regimes."""

    def __init__(
        self, volatility_percentile: float = 0.95, enable_momentum: bool = True
    ) -> None:
        self.regime = RegimeClassifier(volatility_percentile)
        self.band_rejection = BandRejectionStrategy(require_flow_agreement=True)
        self.momentum = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.enable_momentum = enable_momentum
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
        """Select the allowed strategy and retain a machine-readable audit."""

        state = self.regime.evaluate(features, as_of)
        self.last_regime = state
        self.last_strategy = "none"
        if state.label in ("VOLATILE", "TRANSITION"):
            self.last_reason = f"regime-{state.label.lower()}-stand-down"
            return None
        trend_strategies: list[tuple[str, BandRejectionStrategy | MomentumStrategy]] = [
            ("band_rejection", self.band_rejection)
        ]
        if self.enable_momentum:
            trend_strategies.append(("momentum", self.momentum))
        strategies = (
            tuple(trend_strategies)
            if state.label == "TREND"
            else (("mean_reversion", self.mean_reversion),)
        )
        for name, strategy in strategies:
            self.last_strategy = name
            signal = strategy.evaluate(symbol, features, as_of, flow)
            if signal is not None:
                self.last_reason = "signal"
                return signal
        self.last_reason = "no-strategy-trigger"
        return None

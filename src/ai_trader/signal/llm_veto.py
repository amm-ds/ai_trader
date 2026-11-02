"""A local-LLM arm restricted to vetoing or down-sizing existing quant entries."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from ai_trader.agents.local_llm import DecisionClient, DecisionSnapshot
from ai_trader.data.flow import FlowWindow
from ai_trader.evaluation.backtester import SignalStrategy
from ai_trader.signal.band_rejection import TradeSignal


class LLMVetoStrategy:
    """Compose a quant strategy with a bounded LLM; never add or enlarge entries."""

    def __init__(self, quant: SignalStrategy, client: DecisionClient) -> None:
        self.quant = quant
        self.client = client
        self.last_regime = None
        self.last_strategy = "micro_edge+llm_veto"
        self.last_reason = "not-evaluated"

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        signal = self.quant.evaluate(symbol, features, as_of, flow)
        self.last_regime = getattr(self.quant, "last_regime", None)
        if signal is None:
            return None
        snapshot = snapshot_for_signal(signal, as_of, self.last_regime)
        decision = self.client.decide(snapshot)
        if decision.fallback or decision.action == "flat" or decision.action != signal.side:
            self.last_reason = "llm-veto"
            return None
        if decision.confidence < 0.75:
            self.last_reason = "llm-down-size"
            return TradeSignal(
                symbol=signal.symbol,
                side=signal.side,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                thesis=signal.thesis + "-llm-half",
                correlation_group=signal.correlation_group,
                trend=signal.trend,
                cascade_confirmed=signal.cascade_confirmed,
                written_trigger=signal.written_trigger,
                conditions=signal.conditions + ("llm-size-multiplier:0.5",),
                size_multiplier=0.5,
            )
        self.last_reason = "llm-pass"
        return signal


def snapshot_for_signal(signal: TradeSignal, as_of: datetime, regime: object) -> DecisionSnapshot:
    """Serialize one exact entry context for deterministic veto replay."""

    return DecisionSnapshot(
        snapshot_id=f"{signal.symbol}-{as_of.isoformat()}",
        symbol=signal.symbol,
        as_of=as_of.isoformat(),
        regime={"state": str(getattr(regime, "label", "UNKNOWN"))},
        model_probability={"confidence": 1.0},
        flow_features={condition.split(":", 1)[0]: condition.split(":", 1)[1] for condition in signal.conditions if ":" in condition},
        governor={"risk_fraction": 0.02, "daily_loss_limit_r": -3.0, "max_positions": 3},
        position={"open": False},
        quant={"side": signal.side, "entry_price": signal.entry_price, "stop_price": signal.stop_price, "target_price": signal.target_price},
    )

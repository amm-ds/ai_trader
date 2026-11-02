"""Deterministic order vetoes and risk-based size caps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A strategy order proposal; all timestamps must be timezone-aware."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    stop_price: float
    thesis: str
    correlation_group: str
    timestamp: datetime
    data_timestamp: datetime
    written_trigger: str | None = None
    expected_slippage_bps: float = 5.0
    observed_slippage_bps: float | None = None


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """The position attributes needed by the governor."""

    symbol: str
    correlation_group: str


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Complete deterministic state for one risk decision."""

    now: datetime
    equity: float
    day_realized_r: float = 0.0
    positions: tuple[PositionSnapshot, ...] = ()
    last_exit_at: dict[str, datetime] = field(default_factory=dict)
    reentry_counts: dict[str, int] = field(default_factory=dict)
    halted: bool = False
    kill_switch: str | None = None


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Hard Phase-0 risk limits."""

    risk_fraction: float = 0.02
    fee_round_trip_bps: float = 12.0
    daily_loss_limit_r: float = -3.0
    cooldown: timedelta = timedelta(hours=4)
    max_reentries: int = 1
    max_positions: int = 3
    max_correlated_positions: int = 2
    evening_bell_hour_utc: int = 19
    min_stop_fraction: float = 0.001
    max_stop_fraction: float = 0.05
    stale_after: timedelta = timedelta(minutes=15)
    max_slippage_multiple: float = 2.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """An allow/veto decision with a quantity that never exceeds the request."""

    allowed: bool
    approved_quantity: float
    reasons: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()


class Governor:
    """Apply hard vetoes and shrink orders to at most 1R risk."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def review(self, intent: OrderIntent, context: RiskContext) -> RiskDecision:
        """Return a fail-closed decision and all independently detected reasons."""

        reasons: list[str] = []
        if not self._timestamps_valid(intent, context.now):
            return RiskDecision(False, 0.0, ("timestamp-invalid",))
        if intent.data_timestamp > context.now or intent.timestamp > context.now:
            reasons.append("future-timestamp")
        if context.now - intent.data_timestamp > self.config.stale_after:
            reasons.append("stale-data")
        if context.kill_switch is not None:
            reasons.append(f"kill-switch:{context.kill_switch}")
        if context.halted:
            reasons.append("global-halt")
        if not math.isfinite(context.equity) or context.equity <= 0:
            reasons.append("invalid-equity")
        if context.day_realized_r <= self.config.daily_loss_limit_r:
            reasons.append("daily-loss-halt")
        if context.now.astimezone(UTC).hour >= self.config.evening_bell_hour_utc:
            reasons.append("evening-bell")
        if any(position.symbol == intent.symbol for position in context.positions):
            reasons.append("no-averaging-down")
        if len(context.positions) >= self.config.max_positions:
            reasons.append("position-cap")
        correlated = sum(
            position.correlation_group == intent.correlation_group
            for position in context.positions
        )
        if correlated >= self.config.max_correlated_positions:
            reasons.append("thesis-correlation-cap")
        reasons.extend(self._reentry_reasons(intent, context))
        reasons.extend(self._order_reasons(intent))

        if reasons:
            return RiskDecision(False, 0.0, tuple(dict.fromkeys(reasons)))

        slippage = intent.expected_slippage_bps / 10_000.0
        fee_per_fill = self.config.fee_round_trip_bps / 20_000.0
        slipped_entry = intent.entry_price * (
            1 + slippage if intent.side == "long" else 1 - slippage
        )
        slipped_stop = intent.stop_price * (
            1 - slippage if intent.side == "long" else 1 + slippage
        )
        stop_loss_per_unit = abs(slipped_entry - slipped_stop) + fee_per_fill * (
            slipped_entry + slipped_stop
        )
        max_quantity = context.equity * self.config.risk_fraction / stop_loss_per_unit
        approved = min(intent.quantity, max_quantity)
        adjustments = ("size-reduced-to-1r",) if approved < intent.quantity else ()
        return RiskDecision(True, approved, adjustments=adjustments)

    def _timestamps_valid(self, intent: OrderIntent, now: datetime) -> bool:
        timestamps = (intent.timestamp, intent.data_timestamp, now)
        return all(item.tzinfo is not None and item.utcoffset() is not None for item in timestamps)

    def _reentry_reasons(self, intent: OrderIntent, context: RiskContext) -> list[str]:
        last_exit = context.last_exit_at.get(intent.symbol)
        if last_exit is None:
            return []
        reasons: list[str] = []
        elapsed = context.now - last_exit
        if timedelta(0) <= elapsed < self.config.cooldown:
            reasons.append("cooldown")
        if context.reentry_counts.get(intent.symbol, 0) >= self.config.max_reentries:
            reasons.append("reentry-cap")
        if not intent.written_trigger or not intent.written_trigger.strip():
            reasons.append("reentry-trigger-required")
        return reasons

    def _order_reasons(self, intent: OrderIntent) -> list[str]:
        reasons: list[str] = []
        numeric = (intent.quantity, intent.entry_price, intent.stop_price)
        if not all(math.isfinite(value) for value in numeric):
            reasons.append("non-finite-order")
            return reasons
        if intent.quantity <= 0 or intent.entry_price <= 0 or intent.stop_price <= 0:
            reasons.append("invalid-order-values")
            return reasons
        correctly_bound = (
            intent.side == "long" and intent.stop_price < intent.entry_price
        ) or (intent.side == "short" and intent.stop_price > intent.entry_price)
        if not correctly_bound:
            reasons.append("stop-binding-mismatch")
        stop_fraction = abs(intent.entry_price - intent.stop_price) / intent.entry_price
        if not self.config.min_stop_fraction <= stop_fraction <= self.config.max_stop_fraction:
            reasons.append("stop-range")
        if (
            intent.observed_slippage_bps is not None
            and intent.observed_slippage_bps
            > intent.expected_slippage_bps * self.config.max_slippage_multiple
        ):
            reasons.append("slippage-kill-switch")
        return reasons

"""Event-driven replay through the production strategy/risk/broker path."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import numpy as np
import pandas as pd

from ai_trader.data.archive import PerSymbolView
from ai_trader.data.flow import FlowWindow
from ai_trader.execution.paper import ClosedTrade, PaperBroker, Position
from ai_trader.features.indicators import add_strategy_features, completed_history
from ai_trader.risk.governor import (
    Governor,
    OrderIntent,
    PositionSnapshot,
    RiskConfig,
    RiskContext,
)
from ai_trader.signal.band_rejection import BandRejectionStrategy, TradeSignal
from ai_trader.signal.exits import (
    TREND_STRATEGIES,
    chandelier_stop,
    profit_ratchet_armed,
    structural_break,
    trend_target,
)

FeatureMap = Mapping[str, Mapping[str, pd.DataFrame]]
_TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """A bounded UTC simulation with fixed initial equity."""

    start: datetime
    end: datetime
    initial_equity: float = 100.0
    risk_fraction: float = 0.02
    flow_window: timedelta = timedelta(minutes=15)
    execution_timeframe: str = "1m"
    structural_exits: bool = False
    swing_exits: bool = False


@dataclass(frozen=True, slots=True)
class EntryAudit:
    """The strategy and regime evidence attached to an actual paper entry."""

    timestamp: datetime
    symbol: str
    side: str
    trend: str
    cascade_confirmed: bool
    quantity: float
    position_id: str = ""
    regime: str = ""
    strategy: str = ""


@dataclass(frozen=True, slots=True)
class DecisionAudit:
    """A machine-readable allow, hold, or veto record."""

    timestamp: datetime
    symbol: str
    action: str
    reasons: tuple[str, ...]
    regime: str = ""
    strategy: str = ""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Trades, decisions, account statistics, and daily realized R."""

    report: dict[str, float | int]
    trades: tuple[ClosedTrade, ...]
    entries: tuple[EntryAudit, ...]
    decisions: tuple[DecisionAudit, ...]
    daily_r: dict[date, float]


class SignalStrategy(Protocol):
    """The common causal strategy interface used by backtest and replay."""

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None: ...


class Backtester:
    """Replay bars chronologically without a separate simulation-only strategy."""

    def __init__(
        self,
        views: Mapping[str, PerSymbolView],
        governor: Governor | None = None,
        strategy: SignalStrategy | None = None,
        features: FeatureMap | None = None,
    ) -> None:
        self.views = dict(views)
        self.governor = governor or Governor(RiskConfig())
        self.strategy = strategy or BandRejectionStrategy()
        self.features = features

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run one session and return its complete behavioral audit."""

        if config.start >= config.end or config.initial_equity <= 0:
            raise ValueError("invalid backtest range or equity")
        if config.execution_timeframe not in _TIMEFRAME_MINUTES:
            raise ValueError("unsupported execution timeframe")
        if config.swing_exits and not config.structural_exits:
            raise ValueError("swing exits require structural exits")
        features = self.features or {
            symbol: {
                timeframe: add_strategy_features(view.bars(timeframe, as_of=config.end))
                for timeframe in ("5m", "15m", "1h", "4h")
            }
            for symbol, view in self.views.items()
        }
        bar_delta = timedelta(minutes=_TIMEFRAME_MINUTES[config.execution_timeframe])
        raw = {
            symbol: _execution_bars(view, config, bar_delta)
            for symbol, view in self.views.items()
        }
        with TemporaryDirectory(prefix="ai-trader-backtest-") as directory:
            broker = PaperBroker(Path(directory) / "positions.json", persist=False)
            entries: list[EntryAudit] = []
            decisions: list[DecisionAudit] = []
            daily_r: dict[date, float] = defaultdict(float)
            last_exit: dict[str, datetime] = {}
            reentries: dict[str, int] = defaultdict(int)
            current_day = config.start.date()
            prices: dict[str, float] = {}
            initial_stops: dict[str, float] = {}
            highest: dict[str, float] = {}
            lowest: dict[str, float] = {}
            trend_positions: set[str] = set()

            for open_time, event_bars in _iter_events(raw):
                close_time = open_time.to_pydatetime() + bar_delta
                if close_time.date() != current_day:
                    current_day = close_time.date()
                    last_exit = {}
                    reentries = defaultdict(int)
                for position in broker.positions:
                    bar = event_bars.get(position.symbol)
                    if bar is not None:
                        highest[position.symbol] = max(highest[position.symbol], float(bar[1]))
                        lowest[position.symbol] = min(lowest[position.symbol], float(bar[2]))
                for symbol, bar in event_bars.items():
                    prices[symbol] = float(bar[3])
                    closed = broker.process_bar(
                        symbol, float(bar[1]), float(bar[2]), close_time
                    )
                    if closed is not None:
                        self._record_close(closed, daily_r, last_exit)
                        _clear_exit_state(symbol, initial_stops, highest, lowest, trend_positions)
                if close_time.hour >= 19:
                    for position in list(broker.positions):
                        if position.symbol not in prices:
                            continue
                        closed = broker.close_position(
                            position.symbol, prices[position.symbol], close_time, "evening-bell"
                        )
                        self._record_close(closed, daily_r, last_exit)
                        _clear_exit_state(
                            position.symbol, initial_stops, highest, lowest, trend_positions
                        )
                    continue
                execution_minutes = close_time.hour * 60 + close_time.minute
                exit_timeframe = config.execution_timeframe if config.swing_exits else "5m"
                if config.structural_exits and execution_minutes % _TIMEFRAME_MINUTES[exit_timeframe] == 0:
                    closed_trends = self._manage_structural_exits(
                        broker,
                        features,
                        close_time,
                        event_bars,
                        config.flow_window,
                        exit_timeframe,
                        config.swing_exits,
                        initial_stops,
                        highest,
                        lowest,
                        trend_positions,
                    )
                    for closed in closed_trends:
                        self._record_close(closed, daily_r, last_exit)
                        _clear_exit_state(
                            closed.symbol, initial_stops, highest, lowest, trend_positions
                        )
                if close_time.minute % 15 != 0:
                    continue
                for symbol in event_bars:
                    if any(position.symbol == symbol for position in broker.positions):
                        continue
                    signal = self.strategy.evaluate(
                        symbol,
                        features[symbol],
                        close_time,
                        self.views[symbol].flow_window(close_time, config.flow_window),
                    )
                    state = getattr(self.strategy, "last_regime", None)
                    regime = str(state.label) if state is not None else ""
                    routed = str(getattr(self.strategy, "last_strategy", ""))
                    if signal is None:
                        if state is not None:
                            reason = str(getattr(self.strategy, "last_reason", "no-signal"))
                            decisions.append(
                                DecisionAudit(
                                    close_time, symbol, "hold", (reason,), regime, routed
                                )
                            )
                        continue
                    day_r = daily_r[current_day]
                    equity = config.initial_equity + sum(
                        trade.net_pnl for trade in broker.closed_trades
                    )
                    intent = self._intent(signal, close_time, equity, config.risk_fraction)
                    context = RiskContext(
                        now=close_time,
                        equity=equity,
                        day_realized_r=day_r,
                        positions=tuple(_snapshot(position) for position in broker.positions),
                        last_exit_at=last_exit,
                        reentry_counts=dict(reentries),
                    )
                    decision = self.governor.review(intent, context)
                    if not decision.allowed:
                        decisions.append(
                            DecisionAudit(
                                close_time, symbol, "veto", decision.reasons, regime, routed
                            )
                        )
                        continue
                    is_trend = routed in TREND_STRATEGIES
                    target = signal.target_price
                    if is_trend:
                        if config.swing_exits:
                            target = signal.target_price
                        elif config.structural_exits:
                            target = trend_target(features[symbol], signal.side, close_time)
                        else:
                            target = None
                    position = broker.open_position(
                        symbol=signal.symbol,
                        side=signal.side,
                        quantity=decision.approved_quantity,
                        price=signal.entry_price,
                        stop_price=signal.stop_price,
                        timestamp=close_time,
                        thesis=signal.thesis,
                        correlation_group=signal.correlation_group,
                        target_price=target,
                    )
                    initial_stops[symbol] = signal.stop_price
                    highest[symbol] = position.entry_price
                    lowest[symbol] = position.entry_price
                    if is_trend and config.structural_exits:
                        trend_positions.add(symbol)
                    if symbol in last_exit:
                        reentries[symbol] += 1
                    entries.append(
                        EntryAudit(
                            close_time,
                            symbol,
                            signal.side,
                            signal.trend,
                            signal.cascade_confirmed,
                            position.quantity,
                            position.position_id,
                            regime,
                            routed,
                        )
                    )
                    decisions.append(
                        DecisionAudit(close_time, symbol, "entry", (), regime, routed)
                    )
            for position in list(broker.positions):
                closed = broker.close_position(
                    position.symbol, prices[position.symbol], config.end, "session-end"
                )
                self._record_close(closed, daily_r, last_exit)
            return BacktestResult(
                report=broker.report(),
                trades=broker.closed_trades,
                entries=tuple(entries),
                decisions=tuple(decisions),
                daily_r=dict(daily_r),
            )

    def _manage_structural_exits(
        self,
        broker: PaperBroker,
        features: FeatureMap,
        close_time: datetime,
        event_bars: Mapping[str, np.ndarray],
        flow_window: timedelta,
        execution_timeframe: str,
        swing_exits: bool,
        initial_stops: dict[str, float],
        highest: dict[str, float],
        lowest: dict[str, float],
        trend_positions: set[str],
    ) -> list[ClosedTrade]:
        """Apply execution-timeframe breaks and never-widen chandelier ratchets."""

        closed: list[ClosedTrade] = []
        for position in broker.positions:
            symbol = position.symbol
            if symbol not in trend_positions or symbol not in event_bars:
                continue
            history = completed_history(features[symbol][execution_timeframe], close_time)
            if history.empty or history.index[-1] != close_time:
                continue
            execution = history.iloc[-1]
            flow = self.views[symbol].flow_window(close_time, flow_window)
            if not swing_exits and structural_break(position.side, execution, flow):
                closed.append(
                    broker.close_position(
                        symbol, float(execution["c"]), close_time, "structural-break"
                    )
                )
                continue
            armed = profit_ratchet_armed(
                position.side,
                position.entry_price,
                initial_stops[symbol],
                highest[symbol],
                lowest[symbol],
            )
            new_stop = chandelier_stop(
                position.side,
                position.entry_price,
                position.stop_price,
                highest[symbol],
                lowest[symbol],
                float(execution.get("atr14", float("nan"))),
                armed,
            )
            close = float(execution["c"])
            crossed = (position.side == "long" and new_stop >= close) or (
                position.side == "short" and new_stop <= close
            )
            if crossed:
                closed.append(
                    broker.close_position(symbol, close, close_time, "chandelier-stop")
                )
            elif new_stop != position.stop_price:
                broker.ratchet_stop(symbol, new_stop)
        return closed

    @staticmethod
    def _intent(
        signal: TradeSignal, timestamp: datetime, equity: float, risk_fraction: float
    ) -> OrderIntent:
        stop_distance = abs(signal.entry_price - signal.stop_price)
        quantity = equity * risk_fraction / stop_distance
        return OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            thesis=signal.thesis,
            correlation_group=signal.correlation_group,
            timestamp=timestamp,
            data_timestamp=timestamp,
            written_trigger=signal.written_trigger,
        )

    @staticmethod
    def _record_close(
        trade: ClosedTrade, daily_r: dict[date, float], last_exit: dict[str, datetime]
    ) -> None:
        daily_r[trade.closed_at.date()] += trade.r_multiple
        last_exit[trade.symbol] = trade.closed_at


def _clear_exit_state(
    symbol: str,
    initial_stops: dict[str, float],
    highest: dict[str, float],
    lowest: dict[str, float],
    trend_positions: set[str],
) -> None:
    initial_stops.pop(symbol, None)
    highest.pop(symbol, None)
    lowest.pop(symbol, None)
    trend_positions.discard(symbol)


def _iter_events(
    raw: Mapping[str, pd.DataFrame],
) -> Iterator[tuple[pd.Timestamp, dict[str, np.ndarray]]]:
    """Merge actual symbol rows chronologically instead of scanning every symbol per tick."""

    def rows(symbol: str, frame: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, str, np.ndarray]]:
        for timestamp, values in zip(frame.index, frame.to_numpy(), strict=True):
            yield timestamp, symbol, values

    merged = heapq.merge(*(rows(symbol, frame) for symbol, frame in raw.items()))
    current: pd.Timestamp | None = None
    event_bars: dict[str, np.ndarray] = {}
    for timestamp, symbol, values in merged:
        if current is not None and timestamp != current:
            yield current, event_bars
            event_bars = {}
        current = timestamp
        event_bars[symbol] = values
    if current is not None:
        yield current, event_bars


def _execution_bars(
    view: PerSymbolView, config: BacktestConfig, bar_delta: timedelta
) -> pd.DataFrame:
    if config.execution_timeframe == "1m":
        return view.one_minute_bars(config.start, config.end)
    frame = view.bars(config.execution_timeframe, as_of=config.end)
    frame = frame.loc[(frame.index > config.start) & (frame.index <= config.end)].copy()
    frame.index = frame.index - bar_delta
    return frame


def _snapshot(position: Position) -> PositionSnapshot:
    return PositionSnapshot(position.symbol, position.correlation_group)

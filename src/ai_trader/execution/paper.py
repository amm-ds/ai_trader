"""File-persisted paper positions with fixed cost and slippage accounting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class Position:
    """An open paper position and its original risk budget."""

    position_id: str
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    stop_price: float
    initial_risk: float
    opened_at: datetime
    thesis: str
    correlation_group: str
    open_fee: float
    target_price: float | None = None


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A closed position with net PnL and R accounting."""

    position_id: str
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    stop_price: float
    initial_risk: float
    opened_at: datetime
    closed_at: datetime
    gross_pnl: float
    fees: float
    net_pnl: float
    r_multiple: float
    thesis: str
    correlation_group: str
    close_reason: str


class PaperBroker:
    """Execute paper fills, auto-close stops, and atomically persist state."""

    def __init__(
        self,
        state_path: str | Path,
        fee_round_trip_bps: float = 12.0,
        slippage_bps: float = 5.0,
        persist: bool = True,
    ) -> None:
        if fee_round_trip_bps < 0 or slippage_bps < 0:
            raise ValueError("fees and slippage cannot be negative")
        self.state_path = Path(state_path)
        self.fee_per_fill = fee_round_trip_bps / 20_000.0
        self.slippage_rate = slippage_bps / 10_000.0
        self.persist = persist
        self._positions: dict[str, Position] = {}
        self._closed: list[ClosedTrade] = []
        if self.persist and self.state_path.exists():
            self._load()

    @property
    def positions(self) -> tuple[Position, ...]:
        """Return open positions in stable insertion order."""

        return tuple(self._positions.values())

    @property
    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        """Return all closed trades."""

        return tuple(self._closed)

    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        stop_price: float,
        timestamp: datetime,
        thesis: str,
        correlation_group: str,
        target_price: float | None = None,
    ) -> Position:
        """Open a position with adverse entry slippage and persisted initial risk."""

        if symbol in self._positions:
            raise ValueError(f"position already open for {symbol}")
        if not all(math.isfinite(value) and value > 0 for value in (quantity, price, stop_price)):
            raise ValueError("quantity and prices must be finite and positive")
        entry = price * (1 + self.slippage_rate if side == "long" else 1 - self.slippage_rate)
        stop_exit = stop_price * (
            1 - self.slippage_rate if side == "long" else 1 + self.slippage_rate
        )
        open_fee = entry * quantity * self.fee_per_fill
        stop_fee = stop_exit * quantity * self.fee_per_fill
        initial_risk = abs(entry - stop_exit) * quantity + open_fee + stop_fee
        if initial_risk <= 0:
            raise ValueError("stop must differ from slipped entry")
        position_id = f"{symbol}-{timestamp.isoformat()}-{len(self._closed) + 1}"
        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry,
            stop_price=stop_price,
            initial_risk=initial_risk,
            opened_at=timestamp,
            thesis=thesis,
            correlation_group=correlation_group,
            open_fee=open_fee,
            target_price=target_price,
        )
        self._positions[symbol] = position
        self._persist()
        return position

    def process_bar(
        self, symbol: str, high: float, low: float, timestamp: datetime
    ) -> ClosedTrade | None:
        """Close at a stop or target, resolving same-bar ambiguity stop-first."""

        position = self._positions.get(symbol)
        if position is None:
            return None
        stopped = (position.side == "long" and low <= position.stop_price) or (
            position.side == "short" and high >= position.stop_price
        )
        if stopped:
            return self.close_position(symbol, position.stop_price, timestamp, "stop")
        if position.target_price is not None:
            targeted = (
                position.side == "long" and high >= position.target_price
            ) or (position.side == "short" and low <= position.target_price)
            if targeted:
                return self.close_position(symbol, position.target_price, timestamp, "target")
        return None

    def ratchet_stop(self, symbol: str, new_stop: float) -> Position:
        """Move a stop toward profit only; widening is structurally rejected."""

        position = self._positions[symbol]
        if not math.isfinite(new_stop) or new_stop <= 0:
            raise ValueError("stop must be finite and positive")
        if (position.side == "long" and new_stop < position.stop_price) or (
            position.side == "short" and new_stop > position.stop_price
        ):
            raise ValueError("stop widening is forbidden")
        updated = replace(position, stop_price=new_stop)
        self._positions[symbol] = updated
        self._persist()
        return updated

    def close_position(
        self, symbol: str, price: float, timestamp: datetime, reason: str = "manual"
    ) -> ClosedTrade:
        """Close at adverse slippage and include both fill fees in net PnL and R."""

        position = self._positions.pop(symbol)
        exit_price = price * (
            1 - self.slippage_rate if position.side == "long" else 1 + self.slippage_rate
        )
        direction = 1 if position.side == "long" else -1
        gross = direction * (exit_price - position.entry_price) * position.quantity
        close_fee = exit_price * position.quantity * self.fee_per_fill
        fees = position.open_fee + close_fee
        net = gross - fees
        trade = ClosedTrade(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            stop_price=position.stop_price,
            initial_risk=position.initial_risk,
            opened_at=position.opened_at,
            closed_at=timestamp,
            gross_pnl=gross,
            fees=fees,
            net_pnl=net,
            r_multiple=net / position.initial_risk,
            thesis=position.thesis,
            correlation_group=position.correlation_group,
            close_reason=reason,
        )
        self._closed.append(trade)
        self._persist()
        return trade

    def close_all(self, prices: dict[str, float], timestamp: datetime, reason: str) -> list[ClosedTrade]:
        """Close every open position in stable symbol order."""

        closed: list[ClosedTrade] = []
        for symbol in list(self._positions):
            closed.append(self.close_position(symbol, prices[symbol], timestamp, reason))
        return closed

    def report(self) -> dict[str, float | int]:
        """Return net account statistics; no profitability claim is implied."""

        wins = [trade for trade in self._closed if trade.net_pnl > 0]
        losses = [trade for trade in self._closed if trade.net_pnl < 0]
        gross_profit = sum(trade.net_pnl for trade in wins)
        gross_loss = -sum(trade.net_pnl for trade in losses)
        return {
            "closed_trades": len(self._closed),
            "open_positions": len(self._positions),
            "net_pnl": sum(trade.net_pnl for trade in self._closed),
            "fees": sum(trade.fees for trade in self._closed),
            "win_rate": len(wins) / len(self._closed) if self._closed else 0.0,
            "expectancy_r": (
                sum(trade.r_multiple for trade in self._closed) / len(self._closed)
                if self._closed
                else 0.0
            ),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        }

    def _persist(self) -> None:
        if not self.persist:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "positions": [_encode(position) for position in self._positions.values()],
            "closed": [_encode(trade) for trade in self._closed],
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def _load(self) -> None:
        payload: Any = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported paper broker state schema")
        for raw in payload["positions"]:
            position = _decode(Position, raw)
            self._positions[position.symbol] = position
        self._closed = [_decode(ClosedTrade, raw) for raw in payload["closed"]]


def _encode(value: Position | ClosedTrade) -> dict[str, Any]:
    result = asdict(value)
    for key, item in result.items():
        if isinstance(item, datetime):
            result[key] = item.isoformat()
    return result


def _decode(cls: type[Position] | type[ClosedTrade], raw: dict[str, Any]) -> Any:
    values = dict(raw)
    for key in ("opened_at", "closed_at"):
        if key in values:
            values[key] = datetime.fromisoformat(values[key])
    return cls(**values)

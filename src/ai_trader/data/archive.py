"""Causal per-symbol access to read-only desk archives."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ai_trader.data.connectors import load_flow, load_klines
from ai_trader.data.flow import FlowWindow

_TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
_RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}


class PerSymbolView:
    """Provide completed OHLCV bars and causal flow windows for one symbol."""

    def __init__(
        self,
        symbol: str,
        klines: pd.DataFrame,
        flow: pd.DataFrame,
        source_timeframe: str = "1m",
    ) -> None:
        if source_timeframe not in _TIMEFRAMES:
            raise ValueError(f"unsupported source timeframe {source_timeframe!r}")
        self.symbol = symbol
        self.source_timeframe = source_timeframe
        self._klines = klines
        self._flow = flow
        self._resampled: dict[str, pd.DataFrame] = {}

    def one_minute_bars(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return raw bars with open times in `[start, end)`."""

        return self._klines.loc[(self._klines.index >= start) & (self._klines.index < end)].copy()

    def bars(self, timeframe: str, as_of: datetime | None = None) -> pd.DataFrame:
        """Return completed bars labeled by their right-exclusive close time."""

        if timeframe not in _TIMEFRAMES:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        if timeframe not in self._resampled:
            self._resampled[timeframe] = self._resample(timeframe)
        result = self._resampled[timeframe]
        if as_of is not None:
            result = result.loc[result.index <= as_of]
        return result.copy()

    def flow_events(self, as_of: datetime | None = None) -> pd.DataFrame:
        """Return a copy of raw flow events, optionally right-closed at `as_of`."""

        return self._flow.loc[self._flow.index <= as_of].copy() if as_of is not None else self._flow.copy()

    def flow_window(self, as_of: datetime, window: timedelta) -> FlowWindow:
        """Aggregate flow in `(as_of - window, as_of]`."""

        if window <= timedelta(0):
            raise ValueError("flow window must be positive")
        start = as_of - window
        events = self._flow.loc[(self._flow.index > start) & (self._flow.index <= as_of)]
        buys = events.loc[events["side"] == "buy", "vol"]
        sells = events.loc[events["side"] == "sell", "vol"]
        return FlowWindow(
            start=start,
            end=as_of,
            events=len(events),
            buy_volume=float(buys.sum()),
            sell_volume=float(sells.sum()),
        )

    def _resample(self, timeframe: str) -> pd.DataFrame:
        if self._klines.empty:
            empty = self._klines.copy()
            empty.index = pd.DatetimeIndex([], tz="UTC", name="t")
            return empty
        source_minutes = _TIMEFRAMES[self.source_timeframe]
        target_minutes = _TIMEFRAMES[timeframe]
        if target_minutes < source_minutes:
            raise ValueError("cannot downsample below the source timeframe")
        if timeframe == self.source_timeframe:
            result = self._klines.copy()
            result.index = result.index + pd.Timedelta(minutes=source_minutes)
            return result
        grouped = self._klines.resample(
            _RULES[timeframe], closed="left", label="right", origin="epoch"
        )
        result = grouped.agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
        )
        complete = grouped["c"].count() == target_minutes // source_minutes
        return result.loc[complete].dropna()


class ArchiveConnector:
    """Locate and load read-only kline and flow archives for desk symbols."""

    def __init__(self, klines_dir: str | Path, flow_dir: str | Path) -> None:
        self.klines_dir = Path(klines_dir)
        self.flow_dir = Path(flow_dir)

    def view(self, symbol: str) -> PerSymbolView:
        """Load one symbol, using an empty flow frame if its archive is absent."""

        kline_path = self.klines_dir / f"{symbol}_1m.jsonl"
        flow_path = self.flow_dir / f"{symbol}.jsonl"
        klines = load_klines(kline_path)
        flow = load_flow(flow_path) if flow_path.exists() else load_flow_from_records()
        return PerSymbolView(symbol, klines, flow)


def load_flow_from_records() -> pd.DataFrame:
    """Return a typed empty flow frame for symbols without flow history."""

    frame = pd.DataFrame({"cls": [], "side": [], "price": [], "vol": []})
    frame.index = pd.DatetimeIndex([], tz="UTC", name="t")
    return frame.astype({"cls": "int64", "side": "str", "price": "float64", "vol": "float64"})

"""Run behavioral session-replay manifests and print a PASS/FAIL table."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ai_trader.data.archive import ArchiveConnector, PerSymbolView
from ai_trader.evaluation.backtester import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    FeatureMap,
    SignalStrategy,
)


@dataclass(frozen=True, slots=True)
class ExpectedBehavior:
    """Safety invariants asserted by a session manifest."""

    min_daily_r: float = -3.0
    zero_counter_trend_entries: bool = True
    zero_cooldown_violations: bool = True
    no_entries_before_15m_cascade: bool = True


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    """A named UTC session and its read-only symbol set."""

    name: str
    session_date: str
    symbols: tuple[str, ...]
    initial_equity: float
    expected: ExpectedBehavior


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Computed invariant counts and the overall session verdict."""

    manifest: ReplayManifest
    backtest: BacktestResult
    minimum_daily_r: float
    counter_trend_entries: int
    cooldown_violations: int
    pre_cascade_entries: int
    passed: bool


def load_manifest(path: str | Path) -> ReplayManifest:
    """Load and validate one JSON replay manifest."""

    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = raw.get("expected", {})
    return ReplayManifest(
        name=str(raw["name"]),
        session_date=str(raw["session_date"]),
        symbols=tuple(str(symbol) for symbol in raw["symbols"]),
        initial_equity=float(raw.get("initial_equity", 100.0)),
        expected=ExpectedBehavior(
            min_daily_r=float(expected.get("min_daily_r", -3.0)),
            zero_counter_trend_entries=bool(expected.get("zero_counter_trend_entries", True)),
            zero_cooldown_violations=bool(expected.get("zero_cooldown_violations", True)),
            no_entries_before_15m_cascade=bool(
                expected.get("no_entries_before_15m_cascade", True)
            ),
        ),
    )


def run_manifest(
    manifest: ReplayManifest,
    data_root: str | Path,
    strategy: SignalStrategy | None = None,
    structural_exits: bool = False,
    swing_exits: bool = False,
    execution_timeframe: str = "1m",
    feature_factory: Callable[[dict[str, PerSymbolView], datetime], FeatureMap] | None = None,
) -> ReplayResult:
    """Replay a manifest against `data_root/{klines,flow}` and check its contract."""

    day = datetime.combine(
        datetime.strptime(manifest.session_date, "%Y-%m-%d").date(),
        time.min,
        tzinfo=UTC,
    )
    root = Path(data_root)
    connector = ArchiveConnector(root / "klines", root / "flow")
    views = {symbol: connector.view(symbol) for symbol in manifest.symbols}
    end = day + timedelta(days=1)
    features = feature_factory(views, end) if feature_factory is not None else None
    backtest = Backtester(views, strategy=strategy, features=features).run(
        BacktestConfig(
            start=day,
            end=day + timedelta(days=1),
            initial_equity=manifest.initial_equity,
            structural_exits=structural_exits,
            swing_exits=swing_exits,
            execution_timeframe=execution_timeframe,
        )
    )
    minimum = min(backtest.daily_r.values(), default=0.0)
    counter_trend = sum(
        (entry.side == "long" and entry.trend != "bull")
        or (entry.side == "short" and entry.trend != "bear")
        for entry in backtest.entries
    )
    cooldown = _cooldown_violations(backtest)
    pre_cascade = sum(not entry.cascade_confirmed for entry in backtest.entries)
    expected = manifest.expected
    passed = (
        minimum >= expected.min_daily_r - 1e-9
        and (counter_trend == 0 or not expected.zero_counter_trend_entries)
        and (cooldown == 0 or not expected.zero_cooldown_violations)
        and (pre_cascade == 0 or not expected.no_entries_before_15m_cascade)
    )
    return ReplayResult(
        manifest,
        backtest,
        minimum,
        counter_trend,
        cooldown,
        pre_cascade,
        passed,
    )


def format_table(results: Sequence[ReplayResult]) -> str:
    """Format stable, pasteable per-session replay results."""

    lines = [
        "SESSION     TRADES  MIN_DAY_R  COUNTER_TREND  COOLDOWN  PRE_CASCADE  RESULT",
        "--------------------------------------------------------------------------",
    ]
    for result in results:
        lines.append(
            f"{result.manifest.session_date}  {len(result.backtest.trades):6d}  "
            f"{result.minimum_daily_r:9.3f}  {result.counter_trend_entries:13d}  "
            f"{result.cooldown_violations:8d}  {result.pre_cascade_entries:11d}  "
            f"{'PASS' if result.passed else 'FAIL'}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; return nonzero if any session violates its manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("TRADING_DESK_DATA"))
    parser.add_argument("--manifest", action="append", type=Path)
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root or TRADING_DESK_DATA is required")
    repo = Path(__file__).resolve().parents[3]
    paths = args.manifest or sorted((repo / "examples" / "replays").glob("*.json"))
    results = [run_manifest(load_manifest(path), args.data_root) for path in paths]
    print(format_table(results))
    return 0 if all(result.passed for result in results) else 1


def _cooldown_violations(result: BacktestResult) -> int:
    violations = 0
    for symbol in {entry.symbol for entry in result.entries}:
        entries = sorted(
            (entry for entry in result.entries if entry.symbol == symbol),
            key=lambda entry: entry.timestamp,
        )
        exits = sorted(trade.closed_at for trade in result.trades if trade.symbol == symbol)
        used_reentries = 0
        current_day = None
        for entry in entries:
            if entry.timestamp.date() != current_day:
                current_day = entry.timestamp.date()
                used_reentries = 0
            previous = [
                timestamp
                for timestamp in exits
                if timestamp < entry.timestamp and timestamp.date() == current_day
            ]
            if not previous:
                continue
            used_reentries += 1
            if entry.timestamp - previous[-1] < timedelta(hours=4) or used_reentries > 1:
                violations += 1
    return violations


if __name__ == "__main__":
    raise SystemExit(main())

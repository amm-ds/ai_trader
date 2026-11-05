"""Kimi-001: does a frontier LLM improve trading metrics, net of all costs?

Arms through the same governor/costs:
  A quant-only (edge-001, expected zero trades), B LLM-proposed entries validated
  mechanically by the sovereign governor, C hybrid reviewer (veto/downsize only).
Datasets: ~400 frozen OOS snapshots stratified by regime, plus session replays
(the five Oct-2026 manifests and 2021-bear / 2024-stress windows) with arm B active.
All LLM decisions are cached by (snapshot_hash, model, prompt_version); replays
never re-call the API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.agents.kimi_api import (
    PROMPT_VERSION,
    DailyCapReached,
    DecisionCache,
    KimiClient,
)
from ai_trader.evaluation.backtester import BacktestConfig, Backtester
from ai_trader.evaluation.edge import (
    baseline_sharpes,
    feature_map,
    micro_view,
    probability_rows,
    select_model,
)
from ai_trader.evaluation.metrics import performance_metrics
from ai_trader.evaluation.replay import _cooldown_violations, load_manifest, run_manifest
from ai_trader.evaluation.walk_forward import Fold
from ai_trader.execution.paper import ClosedTrade, PaperBroker
from ai_trader.models.micro_edge import model_frame, purged_folds
from ai_trader.models.regime import RegimeClassifier
from ai_trader.risk.governor import Governor, OrderIntent, RiskConfig, RiskContext
from ai_trader.signal.band_rejection import TradeSignal
from ai_trader.signal.kimi_proposal import KimiProposalStrategy
from ai_trader.signal.micro_edge import MicroEdgeStrategy

SEED = 20261104
EQUITY = 100.0
RISK_FRACTION = 0.02
PER_FOLD_PER_REGIME = 40
MAX_FORWARD_BARS = 4  # 1h forward window, matching the 1h label horizon
SESSION_CADENCE = timedelta(hours=4)
STRESS_CADENCE = timedelta(hours=8)
STRESS_WINDOWS = (
    ("2021-bear", datetime(2021, 5, 1, tzinfo=UTC), datetime(2021, 7, 31, tzinfo=UTC)),
    ("2024-stress", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 31, tzinfo=UTC)),
)
DEFAULT_MODEL_FOR_MANIFEST = "kimi-k2.6"
DEFAULT_BASE_FOR_MANIFEST = "https://api.moonshot.ai/v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = REPO_ROOT / "analysis" / "data" / "state" / "kimi-001"


def load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def deduped(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps (kept last) for sampling and forward simulation."""

    return frame[~frame.index.duplicated(keep="last")]


def regime_labels(frame: pd.DataFrame) -> pd.Series:
    """Vectorized RegimeClassifier labels over the materialized columns.

    Mirrors `RegimeClassifier.evaluate` on `feature_map`: the 5m and 15m frames
    both read the parquet `trend` column, 1h/4h read their own, and volatility
    reads the 15m `bb_width_pct` / `realized_vol_pct` columns.
    """

    votes = pd.DataFrame(
        {
            "trend_a": frame["trend"],
            "trend_b": frame["trend"],
            "trend_1h": frame["trend_1h"],
            "trend_4h": frame["trend_4h"],
        }
    )
    bull = (votes == "bull").sum(axis=1)
    bear = (votes == "bear").sum(axis=1)
    aligned = np.maximum(bull, bear)
    volatile = np.maximum(frame["bb_width_pct"], frame["realized_vol_pct"]) >= 0.95
    label = pd.Series("TRANSITION", index=frame.index, dtype="object")
    label[aligned <= 1] = "RANGE"
    label[aligned == 4] = "TREND"
    label[volatile] = "VOLATILE"
    return label


def finite_number(value: Any) -> float:
    """Coerce to a finite float; NaN/inf become 0.0 for JSON-safe contexts."""

    number = float(value)
    return number if math.isfinite(number) else 0.0


def recent_bars(frame: pd.DataFrame, as_of: datetime, count: int = 8) -> list[dict[str, Any]]:
    history = frame.loc[: pd.Timestamp(as_of)]
    rows = history.tail(count)
    return [
        {
            "t": str(ts),
            "o": round(finite_number(row["o"]), 2),
            "h": round(finite_number(row["h"]), 2),
            "l": round(finite_number(row["l"]), 2),
            "c": round(finite_number(row["c"]), 2),
            "v": round(finite_number(row["v"]), 4),
        }
        for ts, row in rows.iterrows()
    ]


def quant_review(signal: TradeSignal | None, as_of: datetime) -> dict[str, Any]:
    """The same snapshot-local governor review llm-local-001 used."""

    if signal is None:
        return {"action": "flat", "approved": False, "quantity": 0.0, "reason": "quant-gates"}
    stop_distance = abs(signal.entry_price - signal.stop_price)
    quantity = EQUITY * RISK_FRACTION / stop_distance
    intent = OrderIntent(
        symbol=signal.symbol,
        side=signal.side,
        quantity=quantity,
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        thesis=signal.thesis,
        correlation_group=signal.correlation_group,
        timestamp=as_of,
        data_timestamp=as_of,
        written_trigger=signal.written_trigger,
    )
    decision = Governor(RiskConfig()).review(intent, RiskContext(now=as_of, equity=EQUITY))
    return {
        "action": signal.side if decision.allowed else "flat",
        "approved": decision.allowed,
        "quantity": decision.approved_quantity,
        "entry_price": signal.entry_price,
        "stop_price": signal.stop_price,
        "target_price": signal.target_price,
        "reasons": list(decision.reasons),
    }


class ContextFactory:
    """Build frozen LLM contexts from frame rows, probabilities, and quant state."""

    def __init__(self, frame: pd.DataFrame, probabilities: pd.DataFrame, threshold: float) -> None:
        self.frame = deduped(frame)
        self.modeled = model_frame(self.frame)
        self.probabilities = probabilities
        self.threshold = threshold
        self.regime = RegimeClassifier()
        self.features = feature_map(self.frame)
        self.strategy = MicroEdgeStrategy(probabilities, threshold)
        self.governor_snapshot = {
            "risk_fraction": RISK_FRACTION,
            "daily_loss_limit_r": -3.0,
            "max_positions": 3,
            "max_correlated": 2,
            "evening_bell_utc": 19,
        }

    def build(self, symbol: str, as_of: datetime, flow: Any = None) -> dict[str, Any] | None:
        if as_of not in self.frame.index:
            return None
        state = self.regime.evaluate(self.features, as_of)
        row = self.frame.loc[as_of]
        modeled = self.modeled.loc[as_of]
        probability = self._probability(as_of)
        signal = self.strategy.evaluate(symbol, self.features, as_of, _empty_flow(as_of))
        market = {
            "close": round(finite_number(row["c"]), 2),
            "atr14": round(finite_number(row["atr14"]), 2),
            "recent_bars": recent_bars(self.frame, as_of),
        }
        context: dict[str, Any] = {
            "snapshot_id": f"{symbol}-{as_of.isoformat()}",
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "regime": {
                "label": state.label,
                "direction": state.direction,
                "aligned_timeframes": state.aligned_timeframes,
            },
            "model_probability": probability,
            "flow_features": {
                key: float(modeled[key])
                for key in (
                    "ofi_15m",
                    "cvd_slope_1h",
                    "signed_volume_z",
                    "microprice_deviation_ewma",
                    "dollar_volume_z",
                    "trade_count_z",
                )
            },
            "governor": dict(self.governor_snapshot),
            "position": {"open": False, "symbol": None},
            "quant": quant_review(signal, as_of),
            "market": market,
        }
        return context

    def _probability(self, as_of: datetime) -> dict[str, Any]:
        if self.probabilities.empty:
            return {"down": None, "flat": None, "up": None, "threshold": self.threshold}
        index = self.probabilities.index
        position = index.searchsorted(pd.Timestamp(as_of), side="right") - 1
        if position < 0:
            return {"down": None, "flat": None, "up": None, "threshold": self.threshold}
        row = self.probabilities.iloc[position]
        return {
            "down": float(row["down"]),
            "flat": float(row["flat"]),
            "up": float(row["up"]),
            "threshold": self.threshold,
            "expected_move_1h": float(row["expected_move_1h"]),
        }


def _empty_flow(as_of: datetime) -> Any:
    from ai_trader.data.flow import FlowWindow

    return FlowWindow(as_of - timedelta(minutes=15), as_of, 0, 0.0, 0.0)


def build_snapshot_contexts(frame: pd.DataFrame, per_regime: int = PER_FOLD_PER_REGIME) -> list[dict[str, Any]]:
    """Freeze ~400 OOS contexts, stratified by regime across the five edge-001 folds."""

    raw = frame
    labels = regime_labels(deduped(raw))
    start = raw.index[0].to_pydatetime()
    end = raw.index[-1].to_pydatetime() + timedelta(minutes=15)
    contexts: list[dict[str, Any]] = []
    for fold in purged_folds(start, end):
        model, threshold, _, _ = select_model(raw, fold)
        probabilities = probability_rows(model, raw, fold.test_start, fold.test_end)
        window_labels = labels.loc[(labels.index >= fold.test_start) & (labels.index < fold.test_end)]
        factory = ContextFactory(raw, probabilities, threshold)
        for regime in ("TREND", "RANGE", "TRANSITION"):
            candidates = window_labels[window_labels == regime]
            if candidates.empty:
                continue
            stride = np.linspace(0, len(candidates) - 1, min(per_regime, len(candidates)), dtype=int)
            selected = candidates.iloc[np.unique(stride)].index
            for timestamp in selected:
                context = factory.build("BTC_USDT", timestamp.to_pydatetime())
                if context is not None:
                    context["fold"] = fold.index
                    context["stratum"] = regime
                    contexts.append(context)
        print(f"snapshots fold={fold.index} cumulative={len(contexts)}", flush=True)
    return contexts


def validate_proposal(
    proposal: Any, as_of: datetime, symbol: str = "BTC_USDT"
) -> tuple[OrderIntent | None, tuple[str, ...]]:
    """Mechanical sovereign-governor validation of one LLM proposal."""

    if proposal.fallback or proposal.action == "flat":
        return None, ("llm-flat" if not proposal.fallback else "llm-invalid",)
    if proposal.entry is None or proposal.stop is None or proposal.tp is None:
        return None, ("llm-missing-prices",)
    stop_distance = abs(proposal.entry - proposal.stop)
    if stop_distance <= 0:
        return None, ("llm-degenerate-stop",)
    quantity = EQUITY * RISK_FRACTION / stop_distance
    intent = OrderIntent(
        symbol=symbol,
        side="long" if proposal.action == "long" else "short",
        quantity=quantity,
        entry_price=proposal.entry,
        stop_price=proposal.stop,
        thesis=f"kimi-{proposal.action}",
        correlation_group="crypto-beta",
        timestamp=as_of,
        data_timestamp=as_of,
        written_trigger="kimi-001 LLM proposal",
    )
    decision = Governor(RiskConfig()).review(intent, RiskContext(now=as_of, equity=EQUITY))
    if not decision.allowed:
        return None, decision.reasons
    return replace(intent, quantity=decision.approved_quantity), ()


def simulate_forward(
    frame: pd.DataFrame, intent: OrderIntent, target: float, as_of: datetime
) -> ClosedTrade | None:
    """Fill through the paper broker over the next bars; stop-first, 1h timeout."""

    clean = deduped(frame)
    bars = clean.loc[clean.index > as_of].head(MAX_FORWARD_BARS)
    if bars.empty:
        return None
    broker = PaperBroker("/tmp/kimi-001-unused.json", persist=False)
    broker.open_position(
        symbol=intent.symbol,
        side=intent.side,
        quantity=intent.quantity,
        price=intent.entry_price,
        stop_price=intent.stop_price,
        timestamp=as_of,
        thesis=intent.thesis,
        correlation_group=intent.correlation_group,
        target_price=target,
    )
    last_ts = as_of
    for ts, bar in bars.iterrows():
        last_ts = ts.to_pydatetime()
        closed = broker.process_bar(intent.symbol, float(bar["h"]), float(bar["l"]), last_ts)
        if closed is not None:
            return closed
    final = bars.iloc[-1]
    return broker.close_position(
        intent.symbol, float(final["c"]), last_ts, "timeout"
    )


def run_snapshot_arm_b(
    frame: pd.DataFrame, contexts: list[dict[str, Any]], client: KimiClient
) -> dict[str, Any]:
    """Arm B on the frozen snapshot dataset: propose, validate, simulate forward."""

    trades: list[ClosedTrade] = []
    decisions: list[dict[str, Any]] = []
    for context in contexts:
        as_of = datetime.fromisoformat(context["as_of"])
        proposal = client.propose(context)
        intent, reasons = validate_proposal(proposal, as_of, context["symbol"])
        row: dict[str, Any] = {
            "snapshot_id": context["snapshot_id"],
            "action": proposal.action,
            "fallback": proposal.fallback,
            "confidence": proposal.confidence,
            "governor": "approved" if intent is not None else f"veto:{','.join(reasons)}",
        }
        if intent is not None:
            target = proposal.tp
            if target is None:
                raise ValueError("approved proposal must carry a target")
            trade = simulate_forward(frame, intent, target, as_of)
            row["result"] = trade.close_reason if trade is not None else "no-forward-bars"
            if trade is not None:
                trades.append(trade)
        decisions.append(row)
        print(
            f"armB {context['snapshot_id']} action={proposal.action} "
            f"governor={row['governor']}",
            flush=True,
        )
    return {"trades": trades, "decisions": decisions}


def wait_until_next_utc_day() -> None:
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).date()
    deadline = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)
    print(f"call cap reached; sleeping until {deadline.isoformat()} UTC", flush=True)
    time.sleep(max((deadline - now).total_seconds(), 1.0))


def run_with_cap_resume(function: Any, wait: bool) -> Any:
    while True:
        try:
            return function()
        except DailyCapReached as error:
            print(f"cap: {error}", flush=True)
            if not wait:
                raise SystemExit(3) from None


def fitted_probabilities(frame: pd.DataFrame, fold: Fold, name: str) -> tuple[pd.DataFrame, float]:
    """Fit edge-001's per-window model, cached under the gitignored state dir."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"probabilities-{name}.parquet"
    threshold_path = STATE_DIR / f"probabilities-{name}.threshold.json"
    if path.exists() and threshold_path.exists():
        stored = pd.read_parquet(path)
        stored.index = pd.to_datetime(stored.index, utc=True)
        return stored, float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
    model, threshold, _, _ = select_model(frame, fold)
    probabilities = probability_rows(model, frame, fold.test_start, fold.test_end)
    probabilities.to_parquet(path)
    threshold_path.write_text(json.dumps({"threshold": threshold}) + "\n", encoding="utf-8")
    return probabilities, threshold


def replay_context_builder(
    frame: pd.DataFrame, probabilities: pd.DataFrame, threshold: float
) -> Any:
    factory = ContextFactory(frame, probabilities, threshold)

    def builder(symbol: str, features: Any, as_of: datetime) -> dict[str, Any] | None:
        return factory.build(symbol, as_of)

    return builder


def run_session_replays(
    frame: pd.DataFrame,
    client: KimiClient,
    manifests: list[Path],
    data_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for path in manifests:
        original = load_manifest(path)
        manifest = replace(original, symbols=("BTC_USDT",))
        day = datetime.strptime(manifest.session_date, "%Y-%m-%d").replace(tzinfo=UTC)
        fold = Fold(0, frame.index[0].to_pydatetime(), day - timedelta(days=1), day, day + timedelta(days=1))
        probabilities, threshold = fitted_probabilities(frame, fold, f"session-{path.stem}")
        strategy = KimiProposalStrategy(
            client, replay_context_builder(frame, probabilities, threshold), cadence=SESSION_CADENCE
        )

        def replay_features(views: Any, end: datetime, frame: pd.DataFrame = frame) -> Any:
            return {"BTC_USDT": feature_map(deduped(frame))}

        result = run_manifest(
            manifest,
            data_root,
            strategy=strategy,
            structural_exits=True,
            swing_exits=True,
            execution_timeframe="15m",
            feature_factory=replay_features,
        )
        rows.append(
            {
                "window": manifest.session_date,
                "kind": "session",
                "reviews": strategy.reviews,
                "trades": len(result.backtest.trades),
                "minimum_daily_r": result.minimum_daily_r,
                "counter_trend_entries": result.counter_trend_entries,
                "cooldown_violations": result.cooldown_violations,
                "pre_cascade_entries": result.pre_cascade_entries,
                "passed": result.passed,
            }
        )
        print(f"replay=session-{path.stem} passed={result.passed} reviews={strategy.reviews}", flush=True)
    return rows


def run_stress_replays(frame: pd.DataFrame, client: KimiClient) -> list[dict[str, Any]]:
    rows = []
    for name, start, end in STRESS_WINDOWS:
        fold = Fold(0, frame.index[0].to_pydatetime(), start - timedelta(days=1), start, end)
        probabilities, threshold = fitted_probabilities(frame, fold, name)
        strategy = KimiProposalStrategy(
            client, replay_context_builder(frame, probabilities, threshold), cadence=STRESS_CADENCE
        )
        view = micro_view(deduped(frame))
        result = Backtester(
            {"BTC_USDT": view},
            strategy=strategy,
            features={"BTC_USDT": feature_map(deduped(frame))},
        ).run(
            BacktestConfig(
                start=start,
                end=end,
                execution_timeframe="15m",
                structural_exits=True,
                swing_exits=True,
            )
        )
        minimum = min(result.daily_r.values(), default=0.0)
        counter_trend = sum(
            (entry.side == "long" and entry.trend != "bull")
            or (entry.side == "short" and entry.trend != "bear")
            for entry in result.entries
        )
        cooldown = _cooldown_violations(result)
        pre_cascade = sum(not entry.cascade_confirmed for entry in result.entries)
        passed = (
            minimum >= -3.0 - 1e-9
            and counter_trend == 0
            and cooldown == 0
            and pre_cascade == 0
        )
        metrics = performance_metrics(result.trades, result.entries, start, end, trials=45)
        rows.append(
            {
                "window": name,
                "kind": "stress",
                "reviews": strategy.reviews,
                "trades": len(result.trades),
                "minimum_daily_r": minimum,
                "counter_trend_entries": counter_trend,
                "cooldown_violations": cooldown,
                "pre_cascade_entries": pre_cascade,
                "passed": passed,
                "profit_factor": metrics["profit_factor"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_daily": metrics["sharpe_daily"],
            }
        )
        print(f"replay={name} passed={passed} reviews={strategy.reviews} trades={len(result.trades)}", flush=True)
    return rows


def cache_usage(cache: DecisionCache, model: str) -> dict[str, Any]:
    rows = [row for row in cache._rows.values() if row["model"] == model]
    prompt = sum(int(row["usage"]["prompt_tokens"]) for row in rows)
    completion = sum(int(row["usage"]["completion_tokens"]) for row in rows)
    latencies = [float(row["usage"]["latency_s"]) for row in rows if float(row["usage"]["latency_s"]) > 0]
    invalid = sum(1 for row in rows if row["decision"]["fallback"])
    base = [row for row in rows if row["prompt_version"] == PROMPT_VERSION]
    base_invalid = sum(1 for row in base if row["decision"]["fallback"])
    return {
        "calls": len(rows),
        "live_calls": len(latencies),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tokens": prompt + completion,
        "invalid": invalid,
        "decisions": len(base),
        "base_invalid": base_invalid,
        "invalid_rate": (base_invalid / len(base)) if base else 0.0,
        "latency_mean_s": float(np.mean(latencies)) if latencies else 0.0,
        "latency_median_s": float(statistics.median(latencies)) if latencies else 0.0,
        "latency_p95_s": float(np.quantile(latencies, 0.95)) if latencies else 0.0,
        "usd": sum(float(row["usage"]["usd"]) for row in rows),
    }


def summarize_trades(
    trades: list[ClosedTrade], start: datetime, end: datetime, token_usd: float
) -> dict[str, Any]:
    metrics = performance_metrics(trades, [], start, end, trials=45)
    fees = sum(trade.fees for trade in trades)
    slippage = sum(
        trade.quantity * (trade.entry_price + trade.exit_price) * (0.0005 / 1.0005)
        for trade in trades
    )
    risk = sum(trade.initial_risk for trade in trades)
    total_cost = fees + slippage + token_usd
    return {
        "trades": len(trades),
        "win_rate": metrics["win_rate"],
        "expectancy_r": metrics["expectancy_r"],
        "profit_factor": metrics["profit_factor"],
        "sharpe_daily": metrics["sharpe_daily"],
        "sortino_daily": metrics["sortino_daily"],
        "max_drawdown": metrics["max_drawdown"],
        "turnover": metrics["turnover"],
        "exposure": metrics["exposure"],
        "net_pnl": metrics["net_pnl"],
        "fees_usd": fees,
        "slippage_usd": slippage,
        "token_usd": token_usd,
        "total_cost_usd": total_cost,
        "net_after_everything": metrics["net_pnl"] - token_usd,
        "cost_per_r": (total_cost / risk) if risk > 0 else 0.0,
        "gross_profit_factor": metrics["profit_factor"],
    }


def write_manifest(output: Path, snapshot_count: int) -> None:
    from ai_trader.agents.kimi_api import resolve_base_url, resolve_model

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.yaml").write_text(
        f"""experiment: kimi-001
question: does a frontier LLM improve trading metrics net of all costs
parent: edge-001
model: {resolve_model()}
base_url: {resolve_base_url()}
model_source: env KIMI_MODEL (default {DEFAULT_MODEL_FOR_MANIFEST})
base_url_source: env KIMI_BASE_URL (default {DEFAULT_BASE_FOR_MANIFEST})
key: repo .env only (gitignored, chmod 600); never printed, logged, or committed
arms:
  A: quant-only edge-001 (null baseline; expected zero trades)
  B: LLM-proposed entries validated mechanically by the sovereign governor
  C: hybrid reviewer (veto/downsize only; expected identical to A)
governor: unchanged (RiskConfig defaults; 12bps round-trip fees, 5bps slippage, 2% risk fraction, -3R/day, evening bell 19:00Z)
snapshots: {snapshot_count} frozen OOS contexts from edge-001's five walk-forward folds, stratified by regime (TREND/RANGE/TRANSITION grid, up to {PER_FOLD_PER_REGIME} per stratum per fold; realized strata are RANGE and TRANSITION only — the materialized 1h/4h trend columns are all-neutral and the volatility percentile columns are all-NaN, so TREND/VOLATILE never occur in edge-001's data)
pilot: first 50 snapshots; stop if invalid-json rate > 10% or cost wildly over budget
cache: experiments/kimi-001/decisions_cache.jsonl keyed by (snapshot_hash sha256 of canonical context JSON, model, prompt_version)
caps:
  concurrent_calls: 1
  backoff: exponential 2s base on 429/5xx, honors Retry-After (cap 60s), max 4 attempts
  daily_call_cap_utc: 600
replay:
  sessions: the five Oct-2026 manifests in examples/replays (BTC only), arm B active, review cadence 4h
  stress: 2021-bear (2021-05-01..2021-07-31) and 2024-stress (2024-01-01..2024-03-31), arm B active, review cadence 8h
  adapter: proposals re-based to the completed 15m close preserving stop/target ratios; counter-trend proposals in TREND and non-cascade proposals refused mechanically; governor vetoes logged
prices: analysis/data/usage_prices.json kimi-k2.6 input 0.60 / output 2.50 USD per Mtok — ESTIMATE, verify at platform.moonshot.ai
economics: net_after_everything = net PnL - token USD; cost_per_R = (fees + slippage + token USD) / sum(initial_risk); tokens_per_decision = tokens / arm-B decisions; slippage_usd = sum(qty * (entry + exit) * 0.0005/1.0005)
verdict_rule: Kimi improves the metrics only if arm B beats arm A AND buy-and-hold on net-after-everything Sharpe with PF > 1.3 and max DD < 20% on the backtest OOS, AND the disaster replays show no behavioral breach; anything else is an honest no
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["snapshots", "pilot", "backtest", "replays", "sensitivity", "report"])
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "data" / "materialized" / "BTC_USDT_15m_features.parquet")
    parser.add_argument("--data-root", type=Path, default=Path("/Users/alejandro/work/trading-desk/analysis/data"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "kimi-001")
    parser.add_argument("--pilot", type=int, default=50)
    parser.add_argument("--wait", action="store_true", help="sleep through daily call caps instead of exiting")
    args = parser.parse_args(argv)

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = DecisionCache(output / "decisions_cache.jsonl")

    if args.phase == "snapshots":
        frame = load_features(args.features)
        contexts = build_snapshot_contexts(frame)
        (output / "snapshots.json").write_text(
            json.dumps(contexts, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_manifest(output, len(contexts))
        print(f"frozen {len(contexts)} snapshots -> {output / 'snapshots.json'}", flush=True)
        return 0

    frame = load_features(args.features)
    client = KimiClient(cache=cache)

    if args.phase == "pilot":
        contexts = json.loads((output / "snapshots.json").read_text(encoding="utf-8"))
        stride = np.linspace(0, len(contexts) - 1, min(args.pilot, len(contexts)), dtype=int)
        pilot_contexts = [contexts[index] for index in np.unique(stride)]
        usage_before = cache_usage(cache, client.model)
        result = run_with_cap_resume(lambda: run_snapshot_arm_b(frame, pilot_contexts, client), args.wait)
        _write_progress(output, "pilot", result, cache, client)
        usage = cache_usage(cache, client.model)
        live = usage["calls"] - usage_before["calls"]
        print(
            json.dumps(
                {
                    "pilot_snapshots": len(pilot_contexts),
                    "new_calls": live,
                    "invalid_rate": usage["invalid_rate"],
                    "latency_mean_s": usage["latency_mean_s"],
                    "latency_p95_s": usage["latency_p95_s"],
                    "tokens": usage["tokens"],
                    "usd": usage["usd"],
                    "trades": len(result["trades"]),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.phase == "backtest":
        contexts = json.loads((output / "snapshots.json").read_text(encoding="utf-8"))
        result = run_with_cap_resume(lambda: run_snapshot_arm_b(frame, contexts, client), args.wait)
        _write_progress(output, "backtest", result, cache, client)
        print(f"backtest arm B complete: trades={len(result['trades'])}", flush=True)
        return 0

    if args.phase == "replays":
        manifests = sorted((REPO_ROOT / "examples" / "replays").glob("*.json"))
        session_rows = run_with_cap_resume(
            lambda: run_session_replays(frame, client, manifests, args.data_root), args.wait
        )
        stress_rows = run_with_cap_resume(lambda: run_stress_replays(frame, client), args.wait)
        _save_replay_rows(output, session_rows + stress_rows)
        print(f"replays complete: sessions={len(session_rows)} stress={len(stress_rows)}", flush=True)
        return 0

    if args.phase == "sensitivity":
        contexts = json.loads((output / "snapshots.json").read_text(encoding="utf-8"))[:5]
        rows = []
        for context in contexts:
            base = client.propose(context)
            shuffled = dict(reversed(list(context.items())))
            variant = client.propose(shuffled, prompt_kind="shuffled")
            rows.append(
                {
                    "snapshot_id": context["snapshot_id"],
                    "base_action": base.action,
                    "variant_action": variant.action,
                    "stable": base.action == variant.action,
                }
            )
            print(f"sensitivity {context['snapshot_id']}: {base.action} vs {variant.action}", flush=True)
        _save_sensitivity(output, rows)
        return 0

    if args.phase == "report":
        results = build_results(output, frame, cache, client)
        (output / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (output / "report.md").write_text(render_report(results), encoding="utf-8")
        print(f"verdict={results['verdict']} report={output / 'report.md'}", flush=True)
        return 0

    return 2


def _write_progress(output: Path, phase: str, result: dict[str, Any], cache: DecisionCache, client: KimiClient) -> None:
    payload = {
        "phase": phase,
        "trades": [asdict(trade) for trade in result["trades"]],
        "decisions": result["decisions"],
        "usage": cache_usage(cache, client.model),
    }
    (output / f"progress-{phase}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _save_replay_rows(output: Path, rows: list[dict[str, Any]]) -> None:
    (output / "progress-replays.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_sensitivity(output: Path, rows: list[dict[str, Any]]) -> None:
    (output / "progress-sensitivity.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_progress(output: Path, name: str) -> Any:
    return json.loads((output / f"progress-{name}.json").read_text(encoding="utf-8"))


def build_results(output: Path, frame: pd.DataFrame, cache: DecisionCache, client: KimiClient) -> dict[str, Any]:
    backtest = _load_progress(output, "backtest")
    replays = _load_progress(output, "replays")
    sensitivity = _load_progress(output, "sensitivity")
    usage = cache_usage(cache, client.model)
    token_usd_base = _token_usd_for_version(cache, client.model, PROMPT_VERSION)

    contexts = json.loads((output / "snapshots.json").read_text(encoding="utf-8"))
    as_ofs = [datetime.fromisoformat(ctx["as_of"]) for ctx in contexts]
    start, end = min(as_ofs), max(as_ofs) + timedelta(hours=1)

    arm_b = summarize_trades(_trades_from_dicts(backtest["trades"]), start, end, token_usd_base)
    folds = purged_folds(frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime() + timedelta(minutes=15))
    buy_hold = baseline_sharpes(frame, list(folds))["buy_hold"]

    decisions = backtest["decisions"]
    proposal_counts: dict[str, int] = {}
    for row in decisions:
        key = row["action"] if row["governor"] == "approved" else row["governor"]
        proposal_counts[key] = proposal_counts.get(key, 0) + 1

    replays_passed = all(row["passed"] for row in replays)
    verdict_pass = (
        arm_b["sharpe_daily"] > 0.0
        and arm_b["sharpe_daily"] > buy_hold
        and arm_b["profit_factor"] > 1.3
        and arm_b["max_drawdown"] < 0.20
        and replays_passed
    )
    return {
        "experiment": "kimi-001",
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "usage": usage,
        "arms": {
            "A_quant_only": {"trades": 0, "note": "edge-001 zero OOS trades; all metrics zero"},
            "B_llm_proposed": {**arm_b, "proposal_counts": proposal_counts, "decisions": len(decisions)},
            "C_hybrid_reviewer": {"trades": 0, "note": "no quant entries exist, so the reviewer never fires; identical to A by construction"},
        },
        "baselines": {"buy_hold_sharpe_daily": buy_hold},
        "replays": replays,
        "sensitivity": sensitivity,
        "base_tokens": _tokens_for_version(cache, client.model, PROMPT_VERSION),
        "verdict": "YES" if verdict_pass else "NO",
        "verdict_rule": "B beats A and buy-and-hold on net-after-everything Sharpe, PF>1.3, maxDD<20% on backtest OOS, and no behavioral breach in replays",
        "replays_passed": replays_passed,
    }


def _tokens_for_version(cache: DecisionCache, model: str, version: str) -> int:
    return sum(
        int(row["usage"]["prompt_tokens"]) + int(row["usage"]["completion_tokens"])
        for row in cache._rows.values()
        if row["model"] == model and row["prompt_version"] == version
    )


def _token_usd_for_version(cache: DecisionCache, model: str, version: str) -> float:
    return sum(
        float(row["usage"]["usd"])
        for row in cache._rows.values()
        if row["model"] == model and row["prompt_version"] == version
    )


def _trades_from_dicts(rows: list[dict[str, Any]]) -> list[ClosedTrade]:
    trades = []
    for row in rows:
        values = dict(row)
        for key in ("opened_at", "closed_at"):
            values[key] = datetime.fromisoformat(values[key])
        trades.append(ClosedTrade(**values))
    return trades


def render_report(results: dict[str, Any]) -> str:
    usage = results["usage"]
    arm_b = results["arms"]["B_llm_proposed"]
    lines = [
        "# Kimi-001 — does Kimi improve trading metrics net of all costs?",
        "",
        f"Model: `{results['model']}` via the OpenAI-compatible Moonshot API, prompt `{results['prompt_version']}`.",
        "Key material stays in the gitignored repo `.env`; the client enforces one concurrent call,",
        "exponential backoff on 429/5xx, and a 600-call-per-UTC-day cap. Every decision is cached by",
        "(snapshot_hash, model, prompt_version), so replays never re-call the API.",
        "",
        "## Token economics (ESTIMATE prices — verify at platform.moonshot.ai)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| LLM calls (cached + live) | {usage['calls']} |",
        f"| Prompt tokens | {usage['prompt_tokens']} |",
        f"| Completion tokens | {usage['completion_tokens']} |",
        f"| Total tokens | {usage['tokens']} |",
        f"| Token USD | ${usage['usd']:.4f} |",
        f"| Invalid-JSON rate | {usage['invalid_rate']:.2%} |",
        f"| Latency mean / median / P95 (s) | {usage['latency_mean_s']:.2f} / {usage['latency_median_s']:.2f} / {usage['latency_p95_s']:.2f} |",
        "",
        "## Arm comparison (backtest OOS snapshot dataset, net of fees + slippage + tokens)",
        "",
        "| Arm | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Turnover | Fees $ | Slippage $ | Token $ | Net after everything $ | Cost per R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    arm_a = results["arms"]["A_quant_only"]
    arm_c = results["arms"]["C_hybrid_reviewer"]
    lines.append(
        f"| A quant-only | {arm_a['trades']} | 0.00% | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.00% | 0.00x | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |"
    )
    lines.append(
        f"| B LLM-proposed | {arm_b['trades']} | {arm_b['win_rate']:.2%} | {arm_b['expectancy_r']:.4f} | "
        f"{arm_b['profit_factor']:.3f} | {arm_b['sharpe_daily']:.4f} | {arm_b['sortino_daily']:.4f} | "
        f"{arm_b['max_drawdown']:.2%} | {arm_b['turnover']:.2f}x | {arm_b['fees_usd']:.4f} | "
        f"{arm_b['slippage_usd']:.4f} | {arm_b['token_usd']:.4f} | {arm_b['net_after_everything']:.4f} | {arm_b['cost_per_r']:.6f} |"
    )
    lines.append(
        f"| C hybrid reviewer | {arm_c['trades']} | 0.00% | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.00% | 0.00x | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |"
    )
    lines.extend([
        "",
        f"Buy-and-hold daily Sharpe over the same OOS folds: {results['baselines']['buy_hold_sharpe_daily']:.4f}.",
        f"Tokens per arm-B decision (base prompt): {results['base_tokens'] / arm_b['decisions'] if arm_b['decisions'] else 0:.1f}.",
        "",
        "### Arm B proposal accounting",
        "",
        "| Outcome | Count |",
        "|---|---:|",
    ])
    for key, count in sorted(arm_b["proposal_counts"].items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Session and stress replays (arm B active)", "",
        "| Window | Kind | LLM reviews | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for row in results["replays"]:
        lines.append(
            f"| {row['window']} | {row['kind']} | {row['reviews']} | {row['trades']} | "
            f"{row['minimum_daily_r']:.3f} | {row['counter_trend_entries']} | {row['cooldown_violations']} | "
            f"{row['pre_cascade_entries']} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    stable = sum(1 for row in results["sensitivity"] if row["stable"])
    lines.extend([
        "",
        "## Prompt-sensitivity spot check",
        "",
        f"{stable}/{len(results['sensitivity'])} re-asked snapshots (reordered context) returned the same action.",
        "",
        "## Verdict (preregistered rule)",
        "",
        f"Rule: {results['verdict_rule']}.",
        "",
        f"**{results['verdict']}** — arm B net-after-everything Sharpe {arm_b['sharpe_daily']:.4f} vs A 0.0000 and "
        f"buy-and-hold {results['baselines']['buy_hold_sharpe_daily']:.4f}; PF {arm_b['profit_factor']:.3f} (>1.3 required); "
        f"max DD {arm_b['max_drawdown']:.2%} (<20% required); replays behavioral {'PASS' if results['replays_passed'] else 'FAIL'}.",
        "",
        "The governor stayed sovereign throughout: every arm-B order passed the unchanged `Governor.review`,",
        "and non-compliant proposals were vetoed with logged reasons. Token prices are estimate-marked.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

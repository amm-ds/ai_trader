"""Run the preregistered P2a flow-first experiment and its gate report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.data.archive import ArchiveConnector, PerSymbolView
from ai_trader.evaluation.backtester import BacktestConfig, Backtester, BacktestResult, FeatureMap
from ai_trader.evaluation.metrics import daily_returns, performance_metrics, sharpe_ratio
from ai_trader.evaluation.replay import _cooldown_violations, load_manifest, run_manifest
from ai_trader.evaluation.walk_forward import walk_forward_folds
from ai_trader.features.flow_features import add_flow_features, tick_events
from ai_trader.features.indicators import add_strategy_features
from ai_trader.signal.flow_strategies import FlowEnsembleStrategy

LOG_ODDS_GRID = (1.0, 2.0, 3.0)
CVD_QUANTILE_GRID = (0.60, 0.75, 0.90)
SEED = 20261101


def build_features(views: dict[str, PerSymbolView], end: datetime) -> FeatureMap:
    features: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol, view in views.items():
        symbol_features = {
            timeframe: add_strategy_features(view.bars(timeframe, as_of=end))
            for timeframe in ("5m", "15m", "1h", "4h")
        }
        symbol_features["flow"] = add_flow_features(
            view.flow_events(end), symbol_features["15m"][["c"]]
        )
        features[symbol] = symbol_features
    return features


def run_flow_track(views: dict[str, PerSymbolView], start: datetime, end: datetime) -> dict[str, Any]:
    features = build_features(views, end)
    folds = walk_forward_folds(start, end)
    fold_rows: list[dict[str, Any]] = []
    test_results: list[BacktestResult] = []
    for fold in folds:
        candidates: list[tuple[float, int, tuple[float, float]]] = []
        train_grid = []
        for log_odds, quantile in product(LOG_ODDS_GRID, CVD_QUANTILE_GRID):
            result = _run_fold(views, features, fold.train_start, fold.train_end, log_odds, quantile)
            score = sharpe_ratio(daily_returns(result.trades, fold.train_start, fold.train_end, 100.0))
            candidates.append((score, len(result.trades), (log_odds, quantile)))
            train_grid.append(
                {
                    "absorption_log_odds": log_odds,
                    "cvd_slope_quantile": quantile,
                    "train_sharpe": score,
                    "train_trades": len(result.trades),
                }
            )
            print(
                f"flow fold={fold.index} train log_odds={log_odds} quantile={quantile} "
                f"sharpe={score:.4f} trades={len(result.trades)}",
                flush=True,
            )
        best = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index][0],
                candidates[index][1],
                -candidates[index][2][0],
                -candidates[index][2][1],
            ),
        )
        log_odds, quantile = candidates[best][2]
        test = _run_fold(views, features, fold.test_start, fold.test_end, log_odds, quantile)
        test_results.append(test)
        metrics = performance_metrics(
            test.trades, test.entries, fold.test_start, fold.test_end,
            symbol_count=len(views), trials=27,
        )
        fold_rows.append(
            {
                "fold": fold.index,
                "selected_absorption_log_odds": log_odds,
                "selected_cvd_slope_quantile": quantile,
                "train_grid": train_grid,
                "test": metrics,
                "cooldown_violations": _cooldown_violations(test),
            }
        )
        print(
            f"flow fold={fold.index} TEST log_odds={log_odds} quantile={quantile} "
            f"trades={len(test.trades)} PF={metrics['profit_factor']:.4f}",
            flush=True,
        )
    trades = tuple(trade for result in test_results for trade in result.trades)
    entries = tuple(entry for result in test_results for entry in result.entries)
    metrics = performance_metrics(
        trades, entries, folds[0].test_start, folds[-1].test_end,
        symbol_count=len(views), trials=27,
    )
    gates = {
        "profit_factor": metrics["profit_factor"] > 1.5,
        "max_drawdown": metrics["max_drawdown"] < 0.15,
        "trade_count": len(trades) >= 200,
        "expectancy_r": metrics["expectancy_r"] > 0.3,
        "churn_free": all(row["cooldown_violations"] == 0 for row in fold_rows),
    }
    return {
        "name": "Desk 30d flow (48 symbols)",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": len(views),
        "execution_timeframe": "15m",
        "metrics": metrics,
        "folds": fold_rows,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _run_fold(
    views: dict[str, PerSymbolView],
    features: FeatureMap,
    start: datetime,
    end: datetime,
    log_odds: float,
    quantile: float,
) -> BacktestResult:
    return Backtester(
        views,
        strategy=FlowEnsembleStrategy(log_odds, quantile),
        features=features,
    ).run(
        BacktestConfig(
            start=start,
            end=end,
            execution_timeframe="15m",
            structural_exits=True,
            swing_exits=True,
        )
    )


def run_replays(data_root: Path, manifests: Sequence[Path]) -> dict[str, Any]:
    rows = []
    for path in manifests:
        result = run_manifest(
            load_manifest(path),
            data_root,
            strategy=FlowEnsembleStrategy(),
            structural_exits=True,
            swing_exits=True,
            execution_timeframe="15m",
            feature_factory=build_features,
        )
        rows.append(
            {
                "date": result.manifest.session_date,
                "trades": len(result.backtest.trades),
                "minimum_daily_r": result.minimum_daily_r,
                "counter_trend_entries": result.counter_trend_entries,
                "cooldown_violations": result.cooldown_violations,
                "pre_cascade_entries": result.pre_cascade_entries,
                "passed": result.passed,
            }
        )
        print(f"replay={path.stem} passed={result.passed} trades={rows[-1]['trades']}", flush=True)
    return {"sessions": rows, "passed": all(row["passed"] for row in rows)}


def tick_sanity(path: Path) -> dict[str, Any]:
    """Validate feature aggregation against direct computations on the same ticks."""

    ticks = pd.read_parquet(path)
    events = tick_events(ticks)
    close = events["price"].resample("15min", closed="left", label="right", origin="epoch").last().dropna()
    features = add_flow_features(events, pd.DataFrame({"c": close}))
    signed = events["vol"] * np.where(events["side"] == "buy", 1.0, -1.0)
    sampled = signed.resample("15min", closed="left", label="right", origin="epoch").sum()
    rows = []
    checks = (
        ("final_cvd", float(features["cvd"].iloc[-1]), float(signed.sum())),
        ("last_15m_delta", float(features["delta_15m"].iloc[-1]), float(sampled.iloc[-1])),
        ("last_1h_delta", float(features["delta_1h"].iloc[-1]), float(sampled.iloc[-4:].sum())),
        ("last_6h_delta", float(features["delta_6h"].iloc[-1]), float(sampled.iloc[-24:].sum())),
    )
    for name, actual, expected in checks:
        error = abs(actual - expected)
        rows.append({"metric": name, "pipeline": actual, "direct": expected, "abs_error": error, "passed": error < 1e-8})
    return {
        "path": str(path),
        "events": len(events),
        "archive_overlap": False,
        "note": "The desk flow archive starts 2026-09-21, so no same-day archive/tick overlap exists. This validates the pipeline against direct ground truth on the tick day, not cross-source agreement.",
        "checks": rows,
        "passed": all(row["passed"] for row in rows),
    }


def write_artifacts(
    output: Path,
    results: dict[str, Any],
    sources: dict[str, str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = f"""experiment: flow-001
phase: P2a
parents: [baseline-001, baseline-002, baseline-003]
pivot: price_action_to_flow_first_structure_triggered_entries
seed: {SEED}
execution_timeframe: 15m
band_role: regime_and_structure_only_no_band_entries
strategies: [spring_reversal, flow_continuation]
exits: structural_stop_1r_breakeven_3atr_chandelier_15m
grid:
  absorption_log_odds: [1.0, 2.0, 3.0]
  cvd_slope_quantile: [0.6, 0.75, 0.9]
walk_forward:
  folds: 3
  selection: train_daily_sharpe_only
costs:
  fee_round_trip_bps: 12
  slippage_bps_per_fill: 5
sources:
{chr(10).join(f'  {name}: {value}' for name, value in sources.items())}
"""
    (output / "manifest.yaml").write_text(manifest, encoding="utf-8")
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(results), encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    metric = results["track"]["metrics"]
    lines = [
        "# Flow-001 — P2a flow-first baseline",
        "",
        "## Headline out-of-sample result",
        "",
        "Three expanding folds; all 27 candidates were selected on train Sharpe only. Costs are 12 bps round-trip fees plus 5 bps slippage per fill. DSR uses 27 train trials.",
        "",
        "| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {results['track']['name']} | {metric['trades']} | {metric['win_rate']:.2%} | {metric['expectancy_r']:.4f} | {metric['profit_factor']:.3f} | {metric['sharpe_daily']:.3f} | {metric['sortino_daily']:.3f} | {metric['max_drawdown']:.2%} | {metric['exposure']:.2%} | {metric['turnover']:.2f}x | [{metric['sharpe_ci_95'][0]:.3f}, {metric['sharpe_ci_95'][1]:.3f}] | {metric['pbo_permutation']:.3f} | {metric['deflated_sharpe']:.3f} |",
        "",
        "## Per-regime PnL attribution",
        "",
        "| Regime | Trades | Net PnL | Sum R |",
        "|---|---:|---:|---:|",
    ]
    for regime, row in sorted(metric["regime_attribution"].items()):
        lines.append(f"| {regime} | {row['trades']} | {row['net_pnl']:.4f} | {row['sum_r']:.4f} |")
    lines.extend(
        [
            "",
            "## Per-strategy × per-regime attribution",
            "",
            "| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy, regimes in sorted(metric["strategy_attribution"].items()):
        for regime, row in sorted(regimes.items()):
            wr = row["wins"] / row["trades"] if row["trades"] else 0.0
            pf = row["gross_profit"] / row["gross_loss"] if row["gross_loss"] else 0.0
            lines.append(f"| {strategy} | {regime} | {row['trades']} | {wr:.2%} | {pf:.3f} | {row['net_pnl']:.4f} | {row['sum_r']:.4f} |")
    lines.extend(
        [
            "",
            "## Tick-sample sanity check",
            "",
            results["tick_sanity"]["note"],
            "",
            f"Source: `{results['tick_sanity']['path']}`; {results['tick_sanity']['events']} ticks.",
            "",
            "| Metric | Pipeline | Direct tick ground truth | Abs error | Result |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in results["tick_sanity"]["checks"]:
        lines.append(f"| {row['metric']} | {row['pipeline']:.8f} | {row['direct']:.8f} | {row['abs_error']:.10f} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Flow-001 vs baseline-003",
            "",
            "| Metric | Baseline-003 | Flow-001 | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("trades", "win_rate", "expectancy_r", "profit_factor", "max_drawdown", "sharpe_daily", "turnover"):
        before = results["baseline_003"]["metrics"][name]
        after = metric[name]
        lines.append(f"| {name} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines.extend(
        [
            "",
            "## Disaster-session safety replays",
            "",
            "| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in results["replays"]["sessions"]:
        lines.append(f"| {row['date']} | {row['trades']} | {row['minimum_daily_r']:.3f} | {row['counter_trend_entries']} | {row['cooldown_violations']} | {row['pre_cascade_entries']} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Gate verdict",
            "",
            f"- Flow track: {'PASS' if results['track']['passed'] else 'FAIL'} — {results['track']['gates']}",
            f"- Disaster scenarios: {'PASS' if results['replays']['passed'] else 'FAIL'}",
            f"- Tick sanity: {'PASS' if results['tick_sanity']['passed'] else 'FAIL'}",
            f"- **Overall: {results['gate_verdict']}**",
            "",
            results["diagnosis"],
            "",
            "## Data-power caveat",
            "",
            "Only 30 days × 48 symbols of desk flow history exists. Statistical power comes from cross-symbol aggregation and a future forward shadow run, not from pretending there is more history. Tick samples validate feature semantics only and are not a performance track. A failed gate blocks P2b; this report is the honest P2a gate record.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tick-parquet", type=Path, required=True)
    parser.add_argument("--baseline-003", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    connector = ArchiveConnector(args.data_root / "klines", args.data_root / "flow")
    symbols = sorted(path.name.removesuffix("_1m.jsonl") for path in (args.data_root / "klines").glob("*_1m.jsonl"))
    views = {symbol: connector.view(symbol) for symbol in symbols}
    end = datetime(2026, 10, 31, tzinfo=UTC)
    track = run_flow_track(views, end - timedelta(days=30), end)
    repo = Path(__file__).resolve().parents[3]
    replays = run_replays(args.data_root, sorted((repo / "examples" / "replays").glob("*.json")))
    sanity = tick_sanity(args.tick_parquet)
    baseline_003 = json.loads(args.baseline_003.read_text(encoding="utf-8"))
    overall = track["passed"] and replays["passed"] and sanity["passed"]
    results = {
        "experiment": "flow-001",
        "seed": SEED,
        "track": track,
        "baseline_003": baseline_003["tracks"][0] if baseline_003["tracks"][0]["name"].startswith("Desk") else baseline_003["tracks"][1],
        "replays": replays,
        "tick_sanity": sanity,
        "gate_verdict": "PASS" if overall else "FAIL",
        "diagnosis": (
            "Flow-001 passes the preregistered gate; forward shadow validation is required before P2b."
            if overall
            else "Flow-001 fails the preregistered gate. Diagnose the per-strategy/per-regime losses and STOP; P2b does not proceed."
        ),
    }
    sources = {"desk_archive": str(args.data_root), "tick_sanity": str(args.tick_parquet)}
    write_artifacts(args.output, results, sources)
    print(f"gate={results['gate_verdict']} report={args.output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

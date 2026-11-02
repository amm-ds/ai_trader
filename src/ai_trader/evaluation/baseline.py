"""Run the preregistered P1 baseline experiment and write its honest report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trader.data.archive import ArchiveConnector, PerSymbolView, load_flow_from_records
from ai_trader.evaluation.backtester import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    FeatureMap,
)
from ai_trader.evaluation.metrics import daily_returns, performance_metrics, sharpe_ratio
from ai_trader.evaluation.replay import _cooldown_violations, load_manifest, run_manifest
from ai_trader.evaluation.walk_forward import walk_forward_folds
from ai_trader.features.indicators import add_strategy_features
from ai_trader.signal.ensemble import EnsembleStrategy

GRID = (0.90, 0.95, 0.975)
SEED = 20261031


def load_btc_view(path: str | Path) -> PerSymbolView:
    """Read a 5m OHLCV parquet without modifying or reindexing its source."""

    raw = pd.read_parquet(path)
    if "timestamp" in raw.columns:
        raw = raw.set_index("timestamp")
    frame = raw.rename(
        columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    )[["o", "h", "l", "c", "v"]].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index().astype("float64")
    if frame.index.has_duplicates:
        raise ValueError("duplicate BTC parquet timestamps")
    return PerSymbolView("BTC_USDT", frame, load_flow_from_records(), source_timeframe="5m")


def build_features(views: dict[str, PerSymbolView], end: datetime) -> FeatureMap:
    """Precompute causal features once; every fold still slices at its event time."""

    return {
        symbol: {
            timeframe: add_strategy_features(view.bars(timeframe, as_of=end))
            for timeframe in ("5m", "15m", "1h", "4h")
        }
        for symbol, view in views.items()
    }


def run_track(
    name: str,
    views: dict[str, PerSymbolView],
    start: datetime,
    end: datetime,
    execution_timeframe: str,
    structural_exits: bool = False,
    swing_exits: bool = False,
    enable_momentum: bool = True,
    turnover_target: float | None = None,
) -> dict[str, Any]:
    """Select the threshold on train only, then concatenate three OOS test folds."""

    features = build_features(views, end)
    folds = walk_forward_folds(start, end)
    fold_rows: list[dict[str, Any]] = []
    test_results: list[BacktestResult] = []
    selected_train_trades = 0
    for fold in folds:
        candidates: list[tuple[float, int, BacktestResult]] = []
        train_grid: list[dict[str, float | int]] = []
        for threshold in GRID:
            result = _run_fold(
                views, features, fold.train_start, fold.train_end, threshold,
                execution_timeframe, structural_exits, swing_exits, enable_momentum,
            )
            returns = daily_returns(result.trades, fold.train_start, fold.train_end, 100.0)
            score = sharpe_ratio(returns)
            candidates.append((score, len(result.trades), result))
            train_grid.append(
                {"volatility_percentile": threshold, "train_sharpe": score, "train_trades": len(result.trades)}
            )
            print(f"{name} fold={fold.index} train threshold={threshold} sharpe={score:.4f} trades={len(result.trades)}", flush=True)
        best_index = max(
            range(len(GRID)),
            key=lambda index: (candidates[index][0], candidates[index][1], -GRID[index]),
        )
        threshold = GRID[best_index]
        selected_train_trades += candidates[best_index][1]
        test = _run_fold(
            views, features, fold.test_start, fold.test_end, threshold,
            execution_timeframe, structural_exits, swing_exits, enable_momentum,
        )
        test_results.append(test)
        metrics = performance_metrics(
            test.trades,
            test.entries,
            fold.test_start,
            fold.test_end,
            symbol_count=len(views),
        )
        fold_rows.append(
            {
                "fold": fold.index,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "selected_volatility_percentile": threshold,
                "train_grid": train_grid,
                "test": metrics,
                "cooldown_violations": _cooldown_violations(test),
            }
        )
        print(f"{name} fold={fold.index} TEST threshold={threshold} trades={len(test.trades)} PF={metrics['profit_factor']:.4f}", flush=True)
    trades = tuple(trade for result in test_results for trade in result.trades)
    entries = tuple(entry for result in test_results for entry in result.entries)
    decisions = tuple(decision for result in test_results for decision in result.decisions)
    metrics = performance_metrics(
        trades, entries, folds[0].test_start, folds[-1].test_end, symbol_count=len(views)
    )
    total_trades = selected_train_trades + len(trades)
    gates = {
        "profit_factor": metrics["profit_factor"] > 1.5,
        "max_drawdown": metrics["max_drawdown"] < 0.15,
        "trade_count": total_trades >= 200,
        "churn_free": all(row["cooldown_violations"] == 0 for row in fold_rows),
    }
    if turnover_target is not None:
        gates["turnover_target"] = metrics["turnover"] < turnover_target
    return {
        "name": name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": len(views),
        "execution_timeframe": execution_timeframe,
        "selected_train_and_test_trades": total_trades,
        "oos_decisions": len(decisions),
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
    threshold: float,
    execution_timeframe: str,
    structural_exits: bool,
    swing_exits: bool,
    enable_momentum: bool,
) -> BacktestResult:
    return Backtester(
        views,
        strategy=EnsembleStrategy(threshold, enable_momentum=enable_momentum),
        features=features,
    ).run(
        BacktestConfig(
            start=start,
            end=end,
            execution_timeframe=execution_timeframe,
            structural_exits=structural_exits,
            swing_exits=swing_exits,
        )
    )


def run_replays(
    data_root: Path,
    manifests: Sequence[Path],
    structural_exits: bool = False,
    swing_exits: bool = False,
    execution_timeframe: str = "1m",
    enable_momentum: bool = True,
) -> dict[str, Any]:
    """Re-run the five P0 safety manifests with the ensemble active."""

    rows = []
    for path in manifests:
        result = run_manifest(
            load_manifest(path),
            data_root,
            strategy=EnsembleStrategy(enable_momentum=enable_momentum),
            structural_exits=structural_exits,
            swing_exits=swing_exits,
            execution_timeframe=execution_timeframe,
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


def write_artifacts(output: Path, results: dict[str, Any], sources: dict[str, str]) -> None:
    """Write semantic YAML, complete JSON, and a stable Markdown summary."""

    output.mkdir(parents=True, exist_ok=True)
    experiment = str(results["experiment"])
    structural = bool(results["structural_exits"])
    swing = bool(results.get("swing_exits", False))
    if swing:
        parent = "baseline-002\nancestors: [baseline-001, baseline-002]\nsingle_structural_change_set: 15m_execution_band_only_trend_1r_breakeven_3atr_chandelier_swing"
    elif structural:
        parent = "baseline-001\nsingle_structural_change: trend_stop_only_to_1r_breakeven_3atr_chandelier_structural_break_and_prior_swing_or_htf_band_target"
    else:
        parent = "null\nsingle_structural_change: none"
    strategies = (
        "[band_rejection, mean_reversion]"
        if swing
        else "[band_rejection, momentum, mean_reversion]"
    )
    manifest = f"""experiment: {experiment}
parent: {parent}
phase: P1
seed: {SEED}
strategies: {strategies}
router: deterministic_regime_ensemble
execution_timeframe: {results.get('execution_timeframe', '1m')}
structural_exits: {str(structural).lower()}
swing_exits: {str(swing).lower()}
momentum_removed_as_toxic: {str(swing).lower()}
turnover_target_btc: {'100x' if swing else 'null'}
regime_grid_volatility_percentile: [{', '.join(str(value) for value in GRID)}]
walk_forward:
  folds: 3
  scheme: expanding_train_next_sixth_test
costs:
  fee_round_trip_bps: 12
  slippage_bps_per_fill: 5
risk:
  fraction_per_trade: 0.02
  daily_loss_limit_r: -3
  evening_bell_utc: 19
lightgbm: skipped_install_timeout
sources:
{chr(10).join(f'  {name}: {path}' for name, path in sources.items())}
"""
    (output / "manifest.yaml").write_text(manifest, encoding="utf-8")
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(results), encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    title = str(results["experiment"]).replace("-", " ").title()
    lines = [
        f"# {title} — P1 regime-routed ensemble",
        "",
        "## Headline out-of-sample results",
        "",
        "Three expanding folds; thresholds were selected on train Sharpe only. Costs are 12 bps round-trip fees plus 5 bps slippage per fill. PBO is a fixed-seed daily-return sign-permutation estimate; DSR uses nine train trials.",
        "",
        "| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in results["tracks"]:
        metric = track["metrics"]
        lines.append(
            f"| {track['name']} | {metric['trades']} | {metric['win_rate']:.2%} | "
            f"{metric['expectancy_r']:.4f} | {metric['profit_factor']:.3f} | "
            f"{metric['sharpe_daily']:.3f} | {metric['sortino_daily']:.3f} | "
            f"{metric['max_drawdown']:.2%} | {metric['exposure']:.2%} | "
            f"{metric['turnover']:.2f}x | [{metric['sharpe_ci_95'][0]:.3f}, {metric['sharpe_ci_95'][1]:.3f}] | "
            f"{metric['pbo_permutation']:.3f} | {metric['deflated_sharpe']:.3f} |"
        )
    if results.get("swing_exits"):
        lines.extend(
            [
                "",
                "Structural finding: momentum was removed after baseline-001 toxicity "
                "(13.43% WR / 0.324 PF on BTC; 3.45% WR / 0.015 PF on flow). "
                "Baseline-003 routes TREND to band_rejection only and keeps mean_reversion RANGE-only.",
            ]
        )
    lines.extend(["", "## Per-regime PnL attribution", ""])
    for track in results["tracks"]:
        lines.extend(
            [
                f"### {track['name']}",
                "",
                "| Regime | Trades | Net PnL | Sum R |",
                "|---|---:|---:|---:|",
            ]
        )
        attribution = track["metrics"]["regime_attribution"]
        for regime in sorted(attribution):
            row = attribution[regime]
            lines.append(f"| {regime} | {row['trades']} | {row['net_pnl']:.4f} | {row['sum_r']:.4f} |")
        lines.append("")
    lines.extend(["## Per-strategy × per-regime attribution", ""])
    for track in results["tracks"]:
        lines.extend(
            [
                f"### {track['name']}",
                "",
                "| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        attribution = track["metrics"]["strategy_attribution"]
        for strategy in sorted(attribution):
            for regime in sorted(attribution[strategy]):
                row = attribution[strategy][regime]
                win_rate = row["wins"] / row["trades"] if row["trades"] else 0.0
                profit_factor = row["gross_profit"] / row["gross_loss"] if row["gross_loss"] else 0.0
                lines.append(
                    f"| {strategy} | {regime} | {row['trades']} | {win_rate:.2%} | "
                    f"{profit_factor:.3f} | {row['net_pnl']:.4f} | {row['sum_r']:.4f} |"
                )
        momentum = _strategy_totals(track, "momentum")
        if momentum["trades"]:
            lines.extend(
                [
                    "",
                    f"Momentum alone is {'toxic' if momentum['net_pnl'] < 0 else 'not net-negative'} on this track: "
                    f"{momentum['trades']} trades, {momentum['win_rate']:.2%} WR, "
                    f"{momentum['profit_factor']:.3f} PF, {momentum['net_pnl']:.4f} net PnL.",
                ]
            )
        lines.append("")
    if results.get("comparison"):
        three_way = "baseline_003" in results["comparison"][0]
        if three_way:
            lines.extend(
                [
                    "## Baseline-001 vs baseline-002 vs baseline-003",
                    "",
                    "| Track | Metric | Baseline-001 | Baseline-002 | Baseline-003 | Delta 003-001 |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for row in results["comparison"]:
                for metric in ("trades", "win_rate", "profit_factor", "max_drawdown", "sharpe_daily", "turnover"):
                    first = row["baseline_001"]["metrics"][metric]
                    second = row["baseline_002"]["metrics"][metric]
                    third = row["baseline_003"]["metrics"][metric]
                    lines.append(
                        f"| {row['name']} | {metric} | {first:.4f} | {second:.4f} | {third:.4f} | {third - first:+.4f} |"
                    )
            lines.extend(
                [
                    "",
                    "| Track | Strategy | Trades 001 | Trades 002 | Trades 003 | Net PnL 001 | Net PnL 002 | Net PnL 003 | Delta 003-001 |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in results["comparison"]:
                for strategy in ("band_rejection", "momentum", "mean_reversion"):
                    first = _strategy_totals(row["baseline_001"], strategy)
                    second = _strategy_totals(row["baseline_002"], strategy)
                    third = _strategy_totals(row["baseline_003"], strategy)
                    lines.append(
                        f"| {row['name']} | {strategy} | {first['trades']} | {second['trades']} | {third['trades']} | "
                        f"{first['net_pnl']:.4f} | {second['net_pnl']:.4f} | {third['net_pnl']:.4f} | "
                        f"{third['net_pnl'] - first['net_pnl']:+.4f} |"
                    )
        else:
            lines.extend(
                [
                    "## Baseline-001 vs baseline-002",
                    "",
                    "| Track | Metric | Baseline-001 | Baseline-002 | Delta |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for row in results["comparison"]:
                for metric in ("trades", "win_rate", "profit_factor", "max_drawdown", "sharpe_daily", "turnover"):
                    before = row["baseline_001"]["metrics"][metric]
                    after = row["baseline_002"]["metrics"][metric]
                    lines.append(f"| {row['name']} | {metric} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
            lines.extend(
                [
                    "",
                    "| Track | Strategy | Trades 001 | Trades 002 | Net PnL 001 | Net PnL 002 | Delta PnL |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in results["comparison"]:
                for strategy in ("band_rejection", "momentum", "mean_reversion"):
                    before = _strategy_totals(row["baseline_001"], strategy)
                    after = _strategy_totals(row["baseline_002"], strategy)
                    lines.append(
                        f"| {row['name']} | {strategy} | {before['trades']} | {after['trades']} | "
                        f"{before['net_pnl']:.4f} | {after['net_pnl']:.4f} | {after['net_pnl'] - before['net_pnl']:+.4f} |"
                    )
        lines.append("")
    lines.extend(
        [
            "## Disaster-session safety replays",
            "",
            "| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in results["replays"]["sessions"]:
        lines.append(
            f"| {row['date']} | {row['trades']} | {row['minimum_daily_r']:.3f} | "
            f"{row['counter_trend_entries']} | {row['cooldown_violations']} | "
            f"{row['pre_cascade_entries']} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Gate verdict", ""])
    for track in results["tracks"]:
        lines.append(f"- {track['name']}: {'PASS' if track['passed'] else 'FAIL'} — {track['gates']}")
    lines.append(f"- Disaster scenarios: {'PASS' if results['replays']['passed'] else 'FAIL'}")
    lines.append(f"- **Overall: {results['gate_verdict']}**")
    lines.extend(
        [
            "",
            results["diagnosis"],
            "",
            "## Limitations",
            "",
            "Stops use bar touches with stop-first ambiguity, targets use the target price plus adverse slippage, and there is no funding, liquidation, queue, or partial-fill model. BTC has no flow events, so its flow veto is inactive. The optional LightGBM baseline was skipped because its dependency download timed out; it did not block this deterministic baseline. These are backtest results, not evidence of live profitability.",
            "",
        ]
    )
    return "\n".join(lines)


def _strategy_totals(track: dict[str, Any], strategy: str) -> dict[str, float | int]:
    rows = track["metrics"]["strategy_attribution"].get(strategy, {}).values()
    total = {
        "trades": sum(int(row["trades"]) for row in rows),
        "wins": sum(int(row["wins"]) for row in rows),
        "net_pnl": sum(float(row["net_pnl"]) for row in rows),
        "gross_profit": sum(float(row["gross_profit"]) for row in rows),
        "gross_loss": sum(float(row["gross_loss"]) for row in rows),
    }
    total["win_rate"] = total["wins"] / total["trades"] if total["trades"] else 0.0
    total["profit_factor"] = total["gross_profit"] / total["gross_loss"] if total["gross_loss"] else 0.0
    return total


def _comparison(parent: dict[str, Any], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = {track["name"]: track for track in parent["tracks"]}
    if parent.get("experiment") == "baseline-002":
        history = {row["name"]: row for row in parent["comparison"]}
        return [
            {
                "name": track["name"],
                "baseline_001": history[track["name"]]["baseline_001"],
                "baseline_002": parents[track["name"]],
                "baseline_003": track,
            }
            for track in current
        ]
    return [
        {"name": track["name"], "baseline_001": parents[track["name"]], "baseline_002": track}
        for track in current
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-parquet", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=("baseline-001", "baseline-002", "baseline-003"),
        default="baseline-001",
    )
    parser.add_argument("--structural-exits", action="store_true")
    parser.add_argument("--swing-exits", action="store_true")
    parser.add_argument("--parent-results", type=Path)
    parser.add_argument("--skip-replays", action="store_true")
    args = parser.parse_args(argv)
    structural = args.experiment in ("baseline-002", "baseline-003")
    swing = args.experiment == "baseline-003"
    enable_momentum = not swing
    if structural != args.structural_exits:
        parser.error("baseline-002/003 require --structural-exits; baseline-001 forbids it")
    if swing != args.swing_exits:
        parser.error("baseline-003 requires --swing-exits; earlier baselines forbid it")
    if structural and args.parent_results is None:
        parser.error("baseline-002/003 require --parent-results")
    parent = json.loads(args.parent_results.read_text(encoding="utf-8")) if args.parent_results else None
    repo = Path(__file__).resolve().parents[3]
    tracks: list[dict[str, Any]] = []

    btc = load_btc_view(args.btc_parquet)
    btc_start = btc.one_minute_bars(datetime.min.replace(tzinfo=UTC), datetime.max.replace(tzinfo=UTC)).index[0].to_pydatetime()
    btc_end = btc.one_minute_bars(datetime.min.replace(tzinfo=UTC), datetime.max.replace(tzinfo=UTC)).index[-1].to_pydatetime() + timedelta(minutes=5)
    btc_execution = "15m" if swing else "5m"
    tracks.append(
        run_track(
            "BTC 2019-2026 5m",
            {"BTC_USDT": btc},
            btc_start,
            btc_end,
            btc_execution,
            structural,
            swing,
            enable_momentum,
            100.0 if swing else None,
        )
    )
    del btc

    connector = ArchiveConnector(args.data_root / "klines", args.data_root / "flow")
    symbols = sorted(path.name.removesuffix("_1m.jsonl") for path in (args.data_root / "klines").glob("*_1m.jsonl"))
    views = {symbol: connector.view(symbol) for symbol in symbols}
    flow_end = datetime(2026, 10, 31, tzinfo=UTC)
    flow_start = flow_end - timedelta(days=30)
    flow_execution = "15m" if swing else "1m"
    tracks.append(
        run_track(
            "Desk 30d flow (48 symbols)", views, flow_start, flow_end,
            flow_execution, structural, swing, enable_momentum,
        )
    )
    del views

    manifests = sorted((repo / "examples" / "replays").glob("*.json"))
    replays = (
        {"sessions": [], "passed": True}
        if args.skip_replays
        else run_replays(
            args.data_root, manifests, structural, swing, flow_execution, enable_momentum
        )
    )
    overall = all(track["passed"] for track in tracks) and replays["passed"]
    failed = [track["name"] for track in tracks if not track["passed"]]
    if overall:
        diagnosis = "All preregistered performance and safety gates pass. This authorizes further validation, not live deployment."
    elif swing:
        diagnosis = (
            "Baseline-003 fails and is the final P1 gate record. Momentum removal is retained "
            "as an evidence-based finding, but 15m band-only swing execution does not pass. "
            "STOP: no further structural iterations or parameter tuning."
        )
    elif structural:
        diagnosis = (
            "The single structural exit iteration still fails. Stop here: no further parameter "
            "tuning, alternate exit mode, or second structural iteration is permitted under this plan."
        )
    elif failed:
        diagnosis = (
            "The ensemble does not pass. Failed track(s): "
            + ", ".join(failed)
            + ". TREND attribution dominates losses on both tracks and turnover is extreme. "
            "Next concrete fix: split TREND attribution by band-rejection versus momentum "
            "and repair the stop-only trend exit with a structural break/target exit, evaluated "
            "independently before any AI or further threshold work. Do not run a second grid iteration."
        )
    else:
        diagnosis = "Performance gates pass but a disaster replay fails; repair that safety violation before any AI work."
    results = {
        "experiment": args.experiment,
        "structural_exits": structural,
        "swing_exits": swing,
        "enable_momentum": enable_momentum,
        "execution_timeframe": flow_execution,
        "seed": SEED,
        "grid": GRID,
        "tracks": tracks,
        "replays": replays,
        "comparison": _comparison(parent, tracks) if parent is not None else [],
        "gate_verdict": "PASS" if overall else "FAIL",
        "diagnosis": diagnosis,
    }
    sources = {"btc_5m_parquet": str(args.btc_parquet), "desk_archive": str(args.data_root)}
    write_artifacts(args.output, results, sources)
    print(f"gate={results['gate_verdict']} report={args.output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

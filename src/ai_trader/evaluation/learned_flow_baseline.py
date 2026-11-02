"""Run flow-002: learned LightGBM on causal microstructure features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.data.archive import ArchiveConnector, PerSymbolView
from ai_trader.evaluation.backtester import BacktestConfig, Backtester, BacktestResult, FeatureMap
from ai_trader.evaluation.flow_baseline import build_features
from ai_trader.evaluation.metrics import performance_metrics
from ai_trader.evaluation.replay import _cooldown_violations, load_manifest, run_manifest
from ai_trader.evaluation.walk_forward import Fold, walk_forward_folds
from ai_trader.models.learned_flow import (
    FEATURE_COLUMNS,
    LearnedFlowModel,
    calibrate_threshold,
    diagnostics,
    make_dataset,
    model_frame,
    probability_frame,
)
from ai_trader.signal.learned_flow import LearnedFlowStrategy

SEED = 20261102


def run_track(views: dict[str, PerSymbolView], start: datetime, end: datetime) -> dict[str, Any]:
    features = build_features(views, end)
    folds = walk_forward_folds(start, end)
    fold_rows = []
    test_results: list[BacktestResult] = []
    for fold in folds:
        model, threshold = train_model(features, start, fold.train_end)
        probabilities = {
            symbol: probability_frame(model, symbol_features, fold.test_start, fold.test_end)
            for symbol, symbol_features in features.items()
        }
        result = _backtest(views, features, probabilities, threshold, fold)
        test_results.append(result)
        model_diagnostics = evaluate_diagnostics(model, features, fold.test_start, fold.test_end)
        metrics = performance_metrics(
            result.trades, result.entries, fold.test_start, fold.test_end,
            symbol_count=len(views), trials=27,
        )
        fold_rows.append(
            {
                "fold": fold.index,
                "probability_threshold": threshold,
                "accuracy_flat_excluded": model_diagnostics["accuracy"],
                "auc_flat_excluded": model_diagnostics["auc"],
                "flat_excluded_rows": model_diagnostics["flat_excluded_rows"],
                "calibration": model_diagnostics["calibration"],
                "feature_importances": model.feature_importances(),
                "test": metrics,
                "cooldown_violations": _cooldown_violations(result),
            }
        )
        print(
            f"flow-002 fold={fold.index} TEST threshold={threshold:.2f} trades={len(result.trades)} "
            f"PF={metrics['profit_factor']:.4f} AUC={model_diagnostics['auc']:.4f}",
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


def train_model(
    features: FeatureMap, start: datetime, cutoff: datetime
) -> tuple[LearnedFlowModel, float]:
    dataset = make_dataset(features, start, cutoff)
    model = LearnedFlowModel(SEED)
    model.fit(dataset)
    threshold = calibrate_threshold(dataset.y, model.predict_proba(dataset.x))
    return model, threshold


def _backtest(
    views: dict[str, PerSymbolView],
    features: FeatureMap,
    probabilities: dict[str, pd.DataFrame],
    threshold: float,
    fold: Fold,
) -> BacktestResult:
    return Backtester(
        views,
        strategy=LearnedFlowStrategy(probabilities, threshold),
        features=features,
    ).run(
        BacktestConfig(
            start=fold.test_start,
            end=fold.test_end,
            execution_timeframe="15m",
            structural_exits=True,
            swing_exits=True,
        )
    )


def evaluate_diagnostics(
    model: LearnedFlowModel, features: FeatureMap, start: datetime, end: datetime
) -> dict[str, Any]:
    labels = []
    probabilities = []
    for symbol_features in features.values():
        frame = model_frame(symbol_features)
        forward = frame["close"].shift(-4)
        returns = forward / frame["close"] - 1
        available = (frame.index >= start) & (frame.index + timedelta(hours=1) <= end)
        x = frame.loc[available, list(FEATURE_COLUMNS)]
        y = pd.Series(
            np.select([returns.loc[available] < -0.001, returns.loc[available] > 0.001], [0, 2], default=1),
            index=x.index,
        )
        labels.append(y)
        probabilities.append(model.predict_proba(x))
    return diagnostics(pd.concat(labels), np.concatenate(probabilities))


def run_replays(data_root: Path, manifests: Sequence[Path], start: datetime) -> dict[str, Any]:
    rows = []
    connector = ArchiveConnector(data_root / "klines", data_root / "flow")
    for path in manifests:
        manifest = load_manifest(path)
        replay_day = datetime.strptime(manifest.session_date, "%Y-%m-%d").replace(tzinfo=UTC)
        views = {symbol: connector.view(symbol) for symbol in manifest.symbols}
        end = replay_day + timedelta(days=1)
        features = build_features(views, end)
        model, threshold = train_model(features, start, replay_day)
        probabilities = {
            symbol: probability_frame(model, features[symbol], replay_day, end)
            for symbol in views
        }

        def replay_features(
            current_views: dict[str, PerSymbolView],
            current_end: datetime,
            features: FeatureMap = features,
        ) -> FeatureMap:
            return features

        result = run_manifest(
            manifest,
            data_root,
            strategy=LearnedFlowStrategy(probabilities, threshold),
            structural_exits=True,
            swing_exits=True,
            execution_timeframe="15m",
            feature_factory=replay_features,
        )
        rows.append(
            {
                "date": manifest.session_date,
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
    output.mkdir(parents=True, exist_ok=True)
    manifest = f"""experiment: flow-002
phase: P2a-learned-final
parents: [baseline-003, flow-001]
pivot: hand_thresholds_to_learned_microstructure
model:
  implementation: lightgbm.Booster_native
  version: 4.7.0
  objective: multiclass_down_flat_up
  n_estimators: 120
  learning_rate: 0.05
  num_leaves: 15
  min_child_samples: 100
  class_weight: balanced
labels:
  horizon: 1h
  flat_band_return: 0.001
probability_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
threshold_selection: train_macro_f1_flat_excluded
walk_forward:
  folds: 3
  pool_symbols: 48
execution_timeframe: 15m
exits: structural_stop_1r_breakeven_3atr_chandelier_15m
costs:
  fee_round_trip_bps: 12
  slippage_bps_per_fill: 5
sources:
{chr(10).join(f'  {name}: {value}' for name, value in sources.items())}
"""
    (output / "manifest.yaml").write_text(manifest, encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(results), encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    metric = results["track"]["metrics"]
    lines = [
        "# Flow-002 — learned flow-features model",
        "",
        "Backend: **native LightGBM 4.7.0 Booster** (installed on the longer-timeout retry; its sklearn wrapper was unavailable because sklearn is absent, so the native API is used with balanced sample weights).",
        "",
        "## Headline out-of-sample result",
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
            "## Model diagnostics",
            "",
            "| Fold | Threshold | Flat-excluded rows | Accuracy | AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in results["track"]["folds"]:
        lines.append(f"| {fold['fold']} | {fold['probability_threshold']:.2f} | {fold['flat_excluded_rows']} | {fold['accuracy_flat_excluded']:.2%} | {fold['auc_flat_excluded']:.4f} |")
    lines.extend(
        [
            "",
            "### Top-10 gain importances by fold",
            "",
            "| Fold | Rank | Feature | Gain share |",
            "|---|---:|---|---:|",
        ]
    )
    for fold in results["track"]["folds"]:
        for rank, row in enumerate(fold["feature_importances"], start=1):
            lines.append(f"| {fold['fold']} | {rank} | {row['feature']} | {row['gain']:.4f} |")
    lines.extend(
        [
            "",
            "### Probability calibration summary",
            "",
            "| Fold | Up-probability bin | Count | Mean probability | Empirical up rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for fold in results["track"]["folds"]:
        for row in fold["calibration"]:
            lines.append(f"| {fold['fold']} | {row['bin']} | {row['count']} | {row['mean_up_probability']:.4f} | {row['empirical_up_rate']:.4f} |")
    lines.extend(
        [
            "",
            "## Flow-002 vs flow-001 and baseline-003",
            "",
            "| Metric | Baseline-003 | Flow-001 | Flow-002 | Delta 002-001 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("trades", "win_rate", "expectancy_r", "profit_factor", "max_drawdown", "sharpe_daily", "turnover"):
        baseline = results["baseline_003"]["metrics"][name]
        flow_001 = results["flow_001"]["metrics"][name]
        current = metric[name]
        lines.append(f"| {name} | {baseline:.4f} | {flow_001:.4f} | {current:.4f} | {current - flow_001:+.4f} |")
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
            f"- **Overall: {results['gate_verdict']}**",
            "",
            results["diagnosis"],
            "",
            "## Data-power caveat",
            "",
            "Only 30 days × 48 symbols of flow history exists. Pooling symbols adds samples but not independent market regimes. Walk-forward tests, flat-excluded diagnostics, and calibration do not establish live edge. Flow-002 is the final experiment in this line; a failed gate stops further model iteration.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--flow-001", type=Path, required=True)
    parser.add_argument("--baseline-003", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    connector = ArchiveConnector(args.data_root / "klines", args.data_root / "flow")
    symbols = sorted(path.name.removesuffix("_1m.jsonl") for path in (args.data_root / "klines").glob("*_1m.jsonl"))
    views = {symbol: connector.view(symbol) for symbol in symbols}
    start = datetime(2026, 10, 1, tzinfo=UTC)
    end = datetime(2026, 10, 31, tzinfo=UTC)
    track = run_track(views, start, end)
    repo = Path(__file__).resolve().parents[3]
    replays = run_replays(args.data_root, sorted((repo / "examples" / "replays").glob("*.json")), start)
    flow_001 = json.loads(args.flow_001.read_text(encoding="utf-8"))
    baseline_003 = json.loads(args.baseline_003.read_text(encoding="utf-8"))
    overall = track["passed"] and replays["passed"]
    results = {
        "experiment": "flow-002",
        "model": "lightgbm-4.7.0",
        "seed": SEED,
        "track": track,
        "flow_001": flow_001["track"],
        "baseline_003": next(track for track in baseline_003["tracks"] if track["name"].startswith("Desk")),
        "replays": replays,
        "gate_verdict": "PASS" if overall else "FAIL",
        "diagnosis": (
            "Flow-002 passes; forward shadow validation is required."
            if overall
            else "Flow-002 fails. The learned model does not repair the flow baseline. STOP: this is the final experiment in the line."
        ),
    }
    write_artifacts(args.output, results, {"desk_archive": str(args.data_root)})
    print(f"gate={results['gate_verdict']} report={args.output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

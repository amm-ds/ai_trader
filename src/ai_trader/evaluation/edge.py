"""Evaluate edge-001 on deep, causal BTC microstructure features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.data.archive import PerSymbolView, load_flow_from_records
from ai_trader.data.ticks import tick_sanity
from ai_trader.evaluation.backtester import BacktestConfig, Backtester, BacktestResult, FeatureMap
from ai_trader.evaluation.metrics import performance_metrics, sharpe_ratio
from ai_trader.evaluation.replay import _cooldown_violations, load_manifest, run_manifest
from ai_trader.evaluation.walk_forward import Fold
from ai_trader.models.micro_edge import (
    FLAT_BANDS,
    THRESHOLD_QUANTILES,
    CalibratedModel,
    evaluate_calibrated,
    fit_calibrated,
    model_frame,
    oof_score,
    purged_folds,
    threshold_from_oof,
)
from ai_trader.signal.micro_edge import MicroEdgeStrategy

SEED = 20261102


def load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def feature_map(frame: pd.DataFrame) -> FeatureMap:
    frames = {}
    for timeframe in ("5m", "15m", "1h", "4h"):
        current = frame.copy()
        if timeframe in ("1h", "4h"):
            current["trend"] = current[f"trend_{timeframe}"]
        frames[timeframe] = current
    return frames


def micro_view(frame: pd.DataFrame) -> PerSymbolView:
    return PerSymbolView("BTC_USDT", frame[["o", "h", "l", "c", "v"]], load_flow_from_records(), source_timeframe="15m")


def select_model(
    frame: pd.DataFrame, fold: Fold
) -> tuple[CalibratedModel, float, float, dict[str, Any]]:
    best: tuple[float, CalibratedModel, float, float, dict[str, Any]] | None = None
    for flat_band in FLAT_BANDS:
        model, oof_y, oof_probabilities = fit_calibrated(frame, fold.train_start, fold.train_end, flat_band, SEED)
        for quantile in THRESHOLD_QUANTILES:
            threshold = threshold_from_oof(oof_y, oof_probabilities, quantile)
            score = oof_score(oof_y, oof_probabilities, threshold)
            candidate = (score, model, threshold, flat_band, {"quantile": quantile, "oof_macro_f1": score})
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise ValueError("no model candidate")
    return best[1], best[2], best[3], best[4]


def probability_rows(model: CalibratedModel, frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    modeled = model_frame(frame)
    selected = modeled.loc[(modeled.index >= start) & (modeled.index < end)]
    probabilities = model.predict_proba(selected)
    result = pd.DataFrame(probabilities, columns=("down", "flat", "up"), index=selected.index)
    result["expected_move_1h"] = selected["expected_move_1h"]
    return result


def run_fold(frame: pd.DataFrame, fold: Fold) -> tuple[BacktestResult, dict[str, Any]]:
    model, threshold, flat_band, selection = select_model(frame, fold)
    probabilities = probability_rows(model, frame, fold.test_start, fold.test_end)
    result = Backtester(
        {"BTC_USDT": micro_view(frame)},
        strategy=MicroEdgeStrategy(probabilities, threshold),
        features={"BTC_USDT": feature_map(frame)},
    ).run(
        BacktestConfig(
            start=fold.test_start,
            end=fold.test_end,
            execution_timeframe="15m",
            structural_exits=True,
            swing_exits=True,
        )
    )
    diagnostics = evaluate_calibrated(model, frame, fold.test_start, fold.test_end, flat_band)
    return result, {
        "fold": fold.index,
        "flat_band": flat_band,
        "probability_threshold": threshold,
        "threshold_quantile": selection["quantile"],
        "oof_macro_f1": selection["oof_macro_f1"],
        "accuracy_flat_excluded": diagnostics["accuracy"],
        "auc_flat_excluded": diagnostics["auc"],
        "flat_excluded_rows": diagnostics["flat_excluded_rows"],
        "calibration": diagnostics["calibration"],
        "feature_importances": model.feature_importances(),
        "cooldown_violations": _cooldown_violations(result),
    }


def run_edge(frame: pd.DataFrame) -> dict[str, Any]:
    start = frame.index[0].to_pydatetime()
    end = frame.index[-1].to_pydatetime() + timedelta(minutes=15)
    folds = purged_folds(start, end)
    fold_rows = []
    results = []
    for fold in folds:
        result, row = run_fold(frame, fold)
        results.append(result)
        row["test"] = performance_metrics(
            result.trades, result.entries, fold.test_start, fold.test_end, trials=45
        )
        fold_rows.append(row)
        print(
            f"edge fold={fold.index} threshold={row['probability_threshold']:.3f} "
            f"trades={len(result.trades)} PF={row['test']['profit_factor']:.4f} AUC={row['auc_flat_excluded']:.4f}",
            flush=True,
        )
    trades = tuple(trade for result in results for trade in result.trades)
    entries = tuple(entry for result in results for entry in result.entries)
    metrics = performance_metrics(trades, entries, folds[0].test_start, folds[-1].test_end, trials=45)
    baselines = baseline_sharpes(frame, folds)
    gates = {
        "profit_factor": metrics["profit_factor"] > 1.5,
        "max_drawdown": metrics["max_drawdown"] < 0.15,
        "trade_count": len(trades) >= 300,
        "expectancy_r": metrics["expectancy_r"] > 0.3,
        "beats_buy_hold_sharpe": metrics["sharpe_daily"] > baselines["buy_hold"],
        "churn_free": all(row["cooldown_violations"] == 0 for row in fold_rows),
    }
    return {
        "name": "BTC deep ticks 2019-2026",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "metrics": metrics,
        "folds": fold_rows,
        "baselines": baselines,
        "gates": gates,
        "passed": all(gates.values()),
    }


def baseline_sharpes(frame: pd.DataFrame, folds: Sequence[Fold]) -> dict[str, float]:
    variants: dict[str, list[float]] = {"buy_hold": [], "ma_cross": [], "quant_only_ofi": []}
    for fold in folds:
        sample = frame.loc[(frame.index >= fold.test_start) & (frame.index < fold.test_end)]
        daily = sample[["c", "ema9", "ma60", "ofi_15m"]].resample("1D").last().dropna()
        returns = daily["c"].pct_change().fillna(0.0)
        buy = returns.copy()
        buy.iloc[0] -= 0.0022
        ma_position = (daily["ema9"] > daily["ma60"]).astype("float64").shift(1).fillna(0.0)
        ma = ma_position * returns - ma_position.diff().abs().fillna(ma_position.abs()) * 0.0022
        ofi_position = np.sign(daily["ofi_15m"]).where(daily["ofi_15m"].abs() > 0.1, 0.0).shift(1).fillna(0.0)
        ofi = ofi_position * returns - ofi_position.diff().abs().fillna(ofi_position.abs()) * 0.0022
        variants["buy_hold"].extend(buy.to_numpy())
        variants["ma_cross"].extend(ma.to_numpy())
        variants["quant_only_ofi"].extend(ofi.to_numpy())
    return {name: sharpe_ratio(np.asarray(values)) for name, values in variants.items()}


def run_stress(frame: pd.DataFrame, windows: Sequence[tuple[str, datetime, datetime]]) -> list[dict[str, Any]]:
    rows = []
    for name, start, end in windows:
        fold = Fold(0, frame.index[0].to_pydatetime(), start - timedelta(days=1), start, end)
        result, row = run_fold(frame, fold)
        metrics = performance_metrics(result.trades, result.entries, start, end, trials=45)
        rows.append({"window": name, "trades": len(result.trades), "profit_factor": metrics["profit_factor"], "max_drawdown": metrics["max_drawdown"], "sharpe": metrics["sharpe_daily"]})
    return rows


def run_replays(frame: pd.DataFrame, manifests: Sequence[Path], data_root: Path) -> dict[str, Any]:
    rows = []
    for path in manifests:
        original = load_manifest(path)
        manifest = replace(original, symbols=("BTC_USDT",))
        day = datetime.strptime(manifest.session_date, "%Y-%m-%d").replace(tzinfo=UTC)
        fold = Fold(0, frame.index[0].to_pydatetime(), day - timedelta(days=1), day, day + timedelta(days=1))
        model, threshold, _, _ = select_model(frame, fold)
        probabilities = probability_rows(model, frame, day, day + timedelta(days=1))

        def replay_features(
            views: dict[str, PerSymbolView], end: datetime, frame: pd.DataFrame = frame
        ) -> FeatureMap:
            return {"BTC_USDT": feature_map(frame)}

        result = run_manifest(
            manifest,
            data_root,
            strategy=MicroEdgeStrategy(probabilities, threshold),
            structural_exits=True,
            swing_exits=True,
            execution_timeframe="15m",
            feature_factory=replay_features,
        )
        rows.append({
            "date": manifest.session_date, "trades": len(result.backtest.trades),
            "minimum_daily_r": result.minimum_daily_r, "counter_trend_entries": result.counter_trend_entries,
            "cooldown_violations": result.cooldown_violations, "pre_cascade_entries": result.pre_cascade_entries,
            "passed": result.passed,
        })
        print(f"replay={path.stem} passed={result.passed} trades={rows[-1]['trades']}", flush=True)
    return {"sessions": rows, "passed": all(row["passed"] for row in rows)}


def write_artifacts(output: Path, results: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = f"""experiment: edge-001
phase: E1-deep-microstructure
parents: [baseline-003, flow-001, flow-002]
model: native_lightgbm_booster_oof_isotonic
calibration: sklearn.IsotonicRegression
folds: 5_expanding_purged_1d_embargo
inner_oof_folds: 3
flat_bands: [0.0008, 0.0010, 0.0015]
threshold_quantiles: [0.6, 0.7, 0.8]
round_trip_cost: 0.0022
execution: 15m
exits: structural_stop_1r_breakeven_3atr_chandelier
feature_scope: [aggressor_pressure, OFI, microprice_EWMA, trade_count, dollar_volume, signed_volume_z, CVD_slope, sweeps, band_regime]
materialization:
  files: {results['materialization']['files']}
  ticks: {results['materialization']['ticks']}
  rows: {results['materialization']['rows']}
  ticks_per_second: {results['materialization']['ticks_per_second']:.0f}
eth_sol: pending
"""
    (output / "manifest.yaml").write_text(manifest, encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(results), encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    track = results["track"]
    metric = track["metrics"]
    lines = [
        "# Edge-001 — deep BTC microstructure",
        "",
        "Native LightGBM + three-fold purged OOF isotonic calibration, five expanding outer folds with one-day embargo. ETH/SOL materialization is pending; this is a complete BTC experiment.",
        "",
        "## Headline out-of-sample result",
        "",
        "| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {track['name']} | {metric['trades']} | {metric['win_rate']:.2%} | {metric['expectancy_r']:.4f} | {metric['profit_factor']:.3f} | {metric['sharpe_daily']:.3f} | {metric['sortino_daily']:.3f} | {metric['max_drawdown']:.2%} | {metric['exposure']:.2%} | {metric['turnover']:.2f}x | [{metric['sharpe_ci_95'][0]:.3f}, {metric['sharpe_ci_95'][1]:.3f}] | {metric['pbo_permutation']:.3f} | {metric['deflated_sharpe']:.3f} |",
        "",
        "## Per-regime attribution",
        "",
        "| Regime | Trades | Net PnL | Sum R |",
        "|---|---:|---:|---:|",
    ]
    for regime, row in sorted(metric["regime_attribution"].items()):
        lines.append(f"| {regime} | {row['trades']} | {row['net_pnl']:.4f} | {row['sum_r']:.4f} |")
    lines.extend([
        "", "## Per-strategy × per-regime attribution", "",
        "| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for strategy, regimes in sorted(metric["strategy_attribution"].items()):
        for regime, row in sorted(regimes.items()):
            wr = row["wins"] / row["trades"] if row["trades"] else 0.0
            pf = row["gross_profit"] / row["gross_loss"] if row["gross_loss"] else 0.0
            lines.append(f"| {strategy} | {regime} | {row['trades']} | {wr:.2%} | {pf:.3f} | {row['net_pnl']:.4f} | {row['sum_r']:.4f} |")
    lines.extend([
        "", "## Model diagnostics", "",
        "| Fold | Flat band | Threshold | Quantile | Flat-excluded rows | Accuracy | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in track["folds"]:
        lines.append(f"| {fold['fold']} | {fold['flat_band']:.4f} | {fold['probability_threshold']:.4f} | {fold['threshold_quantile']:.2f} | {fold['flat_excluded_rows']} | {fold['accuracy_flat_excluded']:.2%} | {fold['auc_flat_excluded']:.4f} |")
    lines.extend(["", "### Top-10 gain importances", "", "| Fold | Rank | Feature | Gain share |", "|---|---:|---|---:|"])
    for fold in track["folds"]:
        for rank, row in enumerate(fold["feature_importances"], start=1):
            lines.append(f"| {fold['fold']} | {rank} | {row['feature']} | {row['gain']:.4f} |")
    lines.extend(["", "### Calibration summary", "", "| Fold | Up-probability bin | Count | Mean probability | Empirical up rate |", "|---|---|---:|---:|---:|"])
    for fold in track["folds"]:
        for row in fold["calibration"]:
            lines.append(f"| {fold['fold']} | {row['bin']} | {row['count']} | {row['mean_up_probability']:.4f} | {row['empirical_up_rate']:.4f} |")
    lines.extend(["", "## Baseline Sharpe comparison", "", "| System | Daily Sharpe |", "|---|---:|"])
    for name, value in track["baselines"].items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.append(f"| edge-001 | {metric['sharpe_daily']:.4f} |")
    lines.append(f"| baseline-003 band system | {results['baseline_003']['metrics']['sharpe_daily']:.4f} |")
    lines.extend(["", "## Stress windows", "", "| Window | Trades | PF | Max DD | Sharpe |", "|---|---:|---:|---:|---:|"])
    for row in results["stress"]:
        lines.append(f"| {row['window']} | {row['trades']} | {row['profit_factor']:.3f} | {row['max_drawdown']:.2%} | {row['sharpe']:.3f} |")
    lines.extend(["", "## Disaster-session safety replays (BTC)", "", "| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |", "|---|---:|---:|---:|---:|---:|---|"])
    for row in results["replays"]["sessions"]:
        lines.append(f"| {row['date']} | {row['trades']} | {row['minimum_daily_r']:.3f} | {row['counter_trend_entries']} | {row['cooldown_violations']} | {row['pre_cascade_entries']} | {'PASS' if row['passed'] else 'FAIL'} |")
    stats = results["materialization"]
    sanity = results["tick_sanity"]
    lines.extend([
        "", "## E0 materialization and tick sanity", "",
        f"BTC files: {stats['files']}; chunks: {stats['chunks']}; ticks: {stats['ticks']}; feature rows: {stats['rows']}; throughput: {stats['files_per_second']:.2f} files/s, {stats['ticks_per_second']:.0f} ticks/s.",
        "",
        "| Check | Absolute error | Result |", "|---|---:|---|",
        f"| CVD | {sanity['cvd_error']:.10f} | {'PASS' if sanity['passed'] else 'FAIL'} |",
        f"| Last 15m delta | {sanity['delta_error']:.10f} | {'PASS' if sanity['passed'] else 'FAIL'} |",
        f"| Imbalance | {sanity['imbalance_error']:.10f} | {'PASS' if sanity['passed'] else 'FAIL'} |",
        "", "## Gate verdict", "",
        f"- Edge track: {'PASS' if track['passed'] else 'FAIL'} — {track['gates']}",
        f"- Disaster scenarios: {'PASS' if results['replays']['passed'] else 'FAIL'}",
        f"- Tick sanity: {'PASS' if sanity['passed'] else 'FAIL'}",
        f"- **Overall: {results['gate_verdict']}**",
        "", results["diagnosis"], "",
        "Ten years of tick history does not guarantee live edge. OOF calibration and deep samples do not create independent future regimes; forward shadow/paper validation remains the arbiter.", "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--materialization-stats", type=Path, required=True)
    parser.add_argument("--tick-sanity-day", type=Path, required=True)
    parser.add_argument("--baseline-003", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = load_features(args.features)
    track = run_edge(frame)
    repo = Path(__file__).resolve().parents[3]
    replays = run_replays(frame, sorted((repo / "examples" / "replays").glob("*.json")), args.data_root)
    stress = run_stress(frame, [
        ("2021-bear", datetime(2021, 5, 1, tzinfo=UTC), datetime(2021, 7, 31, tzinfo=UTC)),
        ("2024-stress", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 31, tzinfo=UTC)),
    ])
    baseline_003 = json.loads(args.baseline_003.read_text())
    sanity = tick_sanity(args.tick_sanity_day)
    overall = track["passed"] and replays["passed"] and sanity["passed"]
    results = {
        "experiment": "edge-001",
        "track": track,
        "materialization": json.loads(args.materialization_stats.read_text()),
        "tick_sanity": sanity,
        "baseline_003": next(track for track in baseline_003["tracks"] if track["name"].startswith("BTC")),
        "stress": stress,
        "replays": replays,
        "gate_verdict": "PASS" if overall else "FAIL",
        "diagnosis": "Edge-001 passes; forward shadow validation is required." if overall else "Edge-001 fails. Deep learned microstructure does not pass the gate. STOP: the AI layer is not authorized.",
    }
    write_artifacts(args.output, results)
    print(f"gate={results['gate_verdict']} report={args.output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

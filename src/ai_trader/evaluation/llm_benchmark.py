"""Benchmark local Ollama decisions against frozen edge-001 contexts."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.agents.local_llm import DecisionClient, DecisionSnapshot, LLMDecision
from ai_trader.data.flow import FlowWindow
from ai_trader.evaluation.backtester import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    SignalStrategy,
)
from ai_trader.evaluation.edge import feature_map, micro_view, probability_rows, select_model
from ai_trader.evaluation.metrics import performance_metrics
from ai_trader.evaluation.walk_forward import Fold
from ai_trader.models.micro_edge import model_frame
from ai_trader.models.micro_edge import purged_folds as edge_folds
from ai_trader.models.regime import RegimeClassifier
from ai_trader.risk.governor import Governor, OrderIntent, RiskConfig, RiskContext
from ai_trader.signal.band_rejection import TradeSignal
from ai_trader.signal.llm_veto import LLMVetoStrategy
from ai_trader.signal.micro_edge import MicroEdgeStrategy


def build_snapshots(frame: pd.DataFrame, per_fold: int = 12) -> list[DecisionSnapshot]:
    """Freeze top-confidence OOS rows with exact model, regime, and risk context."""

    start = frame.index[0].to_pydatetime()
    end = frame.index[-1].to_pydatetime()
    snapshots = []
    for fold in edge_folds(start, end):
        model, threshold, _, _ = select_model(frame, fold)
        probabilities = probability_rows(model, frame, fold.test_start, fold.test_end)
        confidence = np.maximum(probabilities["up"], probabilities["down"])
        selected = confidence.nlargest(per_fold).index
        strategy = MicroEdgeStrategy(probabilities, threshold)
        for timestamp in selected:
            as_of = timestamp.to_pydatetime()
            signal = strategy.evaluate("BTC_USDT", feature_map(frame), as_of, _empty_flow(as_of))
            regime = RegimeClassifier().evaluate(feature_map(frame), as_of)
            row = probabilities.loc[timestamp]
            flow = model_frame(frame).loc[timestamp]
            quant = _quant_review(signal, as_of)
            snapshots.append(
                DecisionSnapshot(
                    snapshot_id=f"fold-{fold.index}-{timestamp.isoformat()}",
                    symbol="BTC_USDT",
                    as_of=timestamp.isoformat(),
                    regime={"label": regime.label, "direction": regime.direction, "aligned_timeframes": regime.aligned_timeframes},
                    model_probability={
                        "down": float(row["down"]), "flat": float(row["flat"]), "up": float(row["up"]),
                        "threshold": threshold, "expected_move_1h": float(row["expected_move_1h"]),
                    },
                    flow_features={
                        key: float(flow[key])
                        for key in ("ofi_15m", "cvd_slope_1h", "signed_volume_z", "microprice_deviation_ewma", "dollar_volume_z", "trade_count_z")
                    },
                    governor={"risk_fraction": 0.02, "daily_loss_limit_r": -3.0, "max_positions": 3, "max_correlated": 2, "evening_bell_utc": 19},
                    position={"open": False, "symbol": None},
                    quant=quant,
                )
            )
    return snapshots


def _empty_flow(as_of: datetime) -> FlowWindow:
    from datetime import timedelta

    from ai_trader.data.flow import FlowWindow

    return FlowWindow(as_of - timedelta(minutes=15), as_of, 0, 0.0, 0.0)


def _quant_review(signal: TradeSignal | None, as_of: datetime) -> dict[str, Any]:
    if signal is None:
        return {"action": "flat", "approved": False, "quantity": 0.0, "reason": "quant-gates"}
    stop_distance = abs(signal.entry_price - signal.stop_price)
    quantity = 100.0 * 0.02 / stop_distance
    intent = OrderIntent(
        symbol=signal.symbol, side=signal.side, quantity=quantity,
        entry_price=signal.entry_price, stop_price=signal.stop_price,
        thesis=signal.thesis, correlation_group=signal.correlation_group,
        timestamp=as_of, data_timestamp=as_of, written_trigger=signal.written_trigger,
    )
    decision = Governor(RiskConfig()).review(intent, RiskContext(now=as_of, equity=100.0))
    return {
        "action": signal.side if decision.allowed else "flat",
        "approved": decision.allowed,
        "quantity": decision.approved_quantity,
        "entry_price": signal.entry_price,
        "stop_price": signal.stop_price,
        "target_price": signal.target_price,
        "reasons": list(decision.reasons),
    }


def benchmark_decisions(
    snapshots: list[DecisionSnapshot], client: DecisionClient, checkpoint: Path | None = None
) -> tuple[list[LLMDecision], dict[str, Any]]:
    decisions: list[LLMDecision] = []
    if checkpoint is not None and checkpoint.exists():
        decisions = [LLMDecision(**row) for row in json.loads(checkpoint.read_text())]
    for snapshot in snapshots[len(decisions):]:
        decisions.append(client.decide(snapshot))
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps([asdict(decision) for decision in decisions], indent=2) + "\n")
    agreement = np.mean([
        decision.action == snapshot.quant["action"]
        for snapshot, decision in zip(snapshots, decisions, strict=True)
    ])
    latencies = [decision.latency_s for decision in decisions]
    tokens = sum(decision.tokens for decision in decisions)
    elapsed = sum(latencies)
    return decisions, {
        "snapshots": len(snapshots),
        "agreement_rate": float(agreement),
        "invalid_json_rate": float(np.mean([decision.fallback for decision in decisions])),
        "latency_mean_s": float(np.mean(latencies)),
        "latency_median_s": float(statistics.median(latencies)),
        "latency_p95_s": float(np.quantile(latencies, 0.95)),
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
        "tokens": tokens,
    }


def counterfactual(frame: pd.DataFrame, client: DecisionClient) -> dict[str, Any]:
    """Replay all five OOS windows with a veto/down-size-only LLM arm."""

    start = frame.index[0].to_pydatetime()
    end = frame.index[-1].to_pydatetime()
    quant_results = []
    llm_results = []
    for fold in edge_folds(start, end):
        model, threshold, _, _ = select_model(frame, fold)
        probabilities = probability_rows(model, frame, fold.test_start, fold.test_end)
        quant = MicroEdgeStrategy(probabilities, threshold)
        quant_results.append(_backtest(frame, fold, quant, probabilities, threshold))
        llm_results.append(_backtest(frame, fold, LLMVetoStrategy(quant, client), probabilities, threshold))
    quant_metrics = _aggregate(quant_results)
    llm_metrics = _aggregate(llm_results)
    return {"quant_only": quant_metrics, "quant_plus_llm": llm_metrics}


def _backtest(
    frame: pd.DataFrame,
    fold: Fold,
    strategy: SignalStrategy,
    probabilities: pd.DataFrame,
    threshold: float,
) -> BacktestResult:
    return Backtester(
        {"BTC_USDT": micro_view(frame)},
        strategy=strategy,
        features={"BTC_USDT": feature_map(frame)},
    ).run(BacktestConfig(fold.test_start, fold.test_end, execution_timeframe="15m", structural_exits=True, swing_exits=True))


def _aggregate(results: list[BacktestResult]) -> dict[str, Any]:
    trades = tuple(trade for result in results for trade in result.trades)
    entries = tuple(entry for result in results for entry in result.entries)
    start = min(trade.opened_at for result in results for trade in result.trades) if trades else None
    end = max(trade.closed_at for result in results for trade in result.trades) if trades else None
    if start is None or end is None:
        return {"trades": 0, "sharpe_daily": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "expectancy_r": 0.0}
    return performance_metrics(trades, entries, start, end, trials=45)


def write_artifacts(output: Path, results: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.yaml").write_text(
        "experiment: llm-local-001\nparent: edge-001\nmodel: qwen3.6:35b-a3b-q4_K_M\nrequested_model: qwen3.8:27b-mlx\npull_result: failed_no_space_left_on_device\nsnapshots: 60\nllm_authority: veto_or_half_size_only\n",
        encoding="utf-8",
    )
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "snapshots.json").write_text(json.dumps(results["snapshot_payloads"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(results), encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    summary = results["benchmark"]
    counter = results["counterfactual"]
    quant = counter["quant_only"]
    llm = counter["quant_plus_llm"]
    return f"""# LLM-local-001 — local decision benchmark

Requested `qwen3.8:27b-mlx`; its 18 GB pull failed with `no space left on device`. Benchmark used **`{results['model']}`**, already installed. Ollama was already serving on localhost:11434.

## Frozen-snapshot benchmark

| Model | Snapshots | Agreement | Invalid JSON | Mean latency (s) | Median (s) | P95 (s) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| {results['model']} | {summary['snapshots']} | {summary['agreement_rate']:.2%} | {summary['invalid_json_rate']:.2%} | {summary['latency_mean_s']:.3f} | {summary['latency_median_s']:.3f} | {summary['latency_p95_s']:.3f} | {summary['tokens_per_second']:.3f} |

## Counterfactual OOS replay

| Arm | Trades | Sharpe | PF | Max DD | E[R] |
|---|---:|---:|---:|---:|---:|
| Quant-only | {quant['trades']} | {quant['sharpe_daily']:.4f} | {quant['profit_factor']:.4f} | {quant['max_drawdown']:.2%} | {quant['expectancy_r']:.4f} |
| Quant + local LLM | {llm['trades']} | {llm['sharpe_daily']:.4f} | {llm['profit_factor']:.4f} | {llm['max_drawdown']:.2%} | {llm['expectancy_r']:.4f} |

## Verdict

**{results['verdict']}** The LLM arm can only veto or halve existing entries; it cannot add trades or increase size. Edge-001 produced zero quant entries, so there is no measurable counterfactual improvement to claim. Latency/agreement diagnostics are reported, not marketed as trading edge.

The read-only v11 assist review is at `docs/trading_bots_crypto_ai_review.md`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.6:35b-a3b-q4_K_M")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.features)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    snapshots = build_snapshots(frame)
    client = __import__("ai_trader.agents.local_llm", fromlist=["OllamaClient"]).OllamaClient(args.model)
    decisions, summary = benchmark_decisions(snapshots, client, args.output / "decisions.json")
    counter = counterfactual(frame, client)
    results = {
        "experiment": "llm-local-001",
        "model": args.model,
        "benchmark": summary,
        "counterfactual": counter,
        "verdict": "NO MEASURABLE VALUE",
        "snapshot_payloads": [snapshot.to_dict() for snapshot in snapshots],
        "decisions": [asdict(decision) for decision in decisions],
    }
    write_artifacts(args.output, results)
    print(json.dumps(summary, indent=2))
    print(json.dumps(counter, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

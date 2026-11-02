# ai_trader

Risk-first, reproducible research infrastructure for MEXC futures. Phase 0 is a
backtesting foundation, **not a live trading bot**: read-only archive connectors,
causal band features, a deterministic risk governor, a paper broker, and an
event-driven replay harness.

The design separates three kinds of intelligence:

- **Statistical:** indicators and, in later phases, validated baseline models.
- **Semantic:** a future Kimi reasoning layer, invoked selectively and evaluated
  against the quant-only baseline.
- **Operational:** deterministic execution and a sovereign risk engine. No model
  or agent can increase exposure or override a veto.

## Architecture

```text
read-only archives -> causal per-symbol views -> band/regime features
                                                        |
                                                        v
                                    regime-routed baseline ensemble
                                                        |
                                                        v
                                      sovereign risk governor (veto/shrink)
                                                        |
                                                        v
                            event-driven backtester -> paper broker -> report
```

The backtester calls the same strategy, governor, and broker implementations a
future shadow/live runner must call. There is no exchange order connector.

## Current results

P1 is complete with **three failed baselines, not a profitable system**. [Baseline-001](experiments/baseline-001/report.md) used stop-only exits (BTC PF 0.402; flow PF 0.182); [baseline-002](experiments/baseline-002/report.md) added 5m structural exits (BTC PF 0.040; flow PF 0.227); [baseline-003](experiments/baseline-003/report.md) removed momentum and used 15m band-only swing execution (BTC PF 0.308, max DD 284.78%, turnover 2697.29x versus the <100x target; flow PF 0.458, max DD 70.52%). Every P1 overall gate is **FAIL**.

P2a [flow-001](experiments/flow-001/report.md) also fails: 123 OOS trades, 34.15% WR, −0.2996 E[R], PF 0.432, max DD 68.06%, and turnover 511.30x. All four tick-ground-truth checks pass, but the 2026-10-08 disaster replay fails at −3.151R. The overall gate is **FAIL**, so P2b does not proceed.

Final learned experiment [flow-002](experiments/flow-002/report.md) uses native LightGBM 4.7.0 and produces only three OOS trades, all stopped (−1.0000 E[R], PF 0.000, max DD 5.96%). Flat-excluded AUC is 0.499–0.529, all five disaster replays pass, but the overall gate is **FAIL**. This is the final experiment in the flow-model line.

Deep-data [edge-001](experiments/edge-001/report.md) materializes 3,683 BTC files (3.63B ticks, 241,871 bars) and uses purged OOF isotonic LightGBM. AUC is 0.526–0.564, but probability plus cost gates produce **zero OOS trades**. Buy-and-hold Sharpe is 0.863 versus 0.000 for edge-001. Tick sanity and all disaster replays pass, but the overall gate is **FAIL**; the AI layer is not authorized. ETH/SOL remain pending.

## Quickstart

```bash
cd ai_trader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check src tests examples
mypy src
pytest
```

Generate a tiny synthetic fixture:

```bash
python examples/generate_synthetic.py --output /tmp/ai_trader_klines.jsonl
```

Replay the five desk sessions against an existing read-only archive (nothing in
the archive is modified):

```bash
python -m ai_trader.evaluation.replay \
  --data-root /path/to/trading-desk/analysis/data
```

Pass one manifest with `--manifest examples/replays/2026-10-08.json` for a
single-session run. Each manifest checks daily loss, trend alignment, cooldowns,
and the completed-15-minute cascade gate.

## Honest limitations

- The first walk-forward baseline fails its performance gates. PBO/deflated
  Sharpe and out-of-sample splits quantify uncertainty; they do not create edge.
- Historical fills are simplified: stop/target fills use their trigger price,
  then apply fixed slippage. No queue, liquidation, funding, or partial-fill
  model is included.
- Reversal, breakout, portfolio optimization, and live/demo execution are not
  implemented. The optional LightGBM baseline was skipped after an install
  timeout; the deterministic ensemble has no ML dependency.
- External desk archives are optional development data and are not redistributed.
  CI uses the small synthetic fixture in `examples/fixtures/`.
- This is research software, not financial advice.

## License

MIT; see [LICENSE](LICENSE).

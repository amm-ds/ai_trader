# Evaluation

The backtester replays source or execution-resampled bars in timestamp order
(one-minute, five-minute, or 15-minute). Stops and targets are processed before new
entries, with stop-first same-bar ambiguity. Strategy reviews occur at completed
15-minute boundaries and can only use feature bars and flow already available at
that instant. Positions are flattened at 19:00Z and at session end.

Paper accounting applies 6 bps fees per fill (12 bps round trip) and 5 bps
adverse slippage per fill. Net PnL divided by initial stop risk (price risk plus
modeled stop-exit costs) gives R, so a modeled stop is -1R.

Session manifests encode behavioral contracts rather than profit targets:
realized daily R must not cross -3R, entries must follow the global trend,
cooldown/re-entry limits must hold, and every entry requires a confirmed 15m
cascade. The replay CLI prints one PASS/FAIL row per session. These are safety
regressions, not evidence of edge.

P1 concatenates three strictly later test folds. Regime thresholds are selected
on train Sharpe only. Reports include daily Sharpe/Sortino, drawdown, exposure,
turnover, fixed-seed block-bootstrap intervals, sign-permutation PBO, deflated
Sharpe, and per-strategy × per-regime attribution. Baseline-001 is the stop-only
parent; baseline-002 changes the 5m trend-exit structure; baseline-003 is the final
15m band-only swing experiment with a <100x BTC turnover target. The three-way
comparison is in `experiments/baseline-003/report.md`.

P2a flow-001 uses the same folds, costs, governor, and 15m swing exits. Its two
parameters form a preregistered 3×3 train-only grid; gates add E[R] > +0.3R and
require at least 200 OOS trades. Real tick samples validate delta/CVD semantics
only, never performance. See `experiments/flow-001/report.md`.

Flow-002 trains one model per fold on pooled symbols, labels 1h forward returns
into down/flat/up with a ±0.1% noise band, and selects probability thresholds by
train macro-F1. Reports include flat-excluded accuracy/AUC, gain importance,
calibration bins, and the same trading/safety gates. See `experiments/flow-002/`.

Edge-001 uses daily materialized BTC ticks, five outer folds with one-day embargo,
three purged inner OOF models for isotonic calibration, and a mandatory
`(2p−1)·E|1h move| > 0.0022` cost gate. Reports add buy-and-hold, MA-cross, OFI
quant-only, 2021-bear, and 2024-stress comparisons. See `experiments/edge-001/`.

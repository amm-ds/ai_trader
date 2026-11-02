# Flow-001 — P2a flow-first baseline

## Headline out-of-sample result

Three expanding folds; all 27 candidates were selected on train Sharpe only. Costs are 12 bps round-trip fees plus 5 bps slippage per fill. DSR uses 27 train trials.

| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Desk 30d flow (48 symbols) | 123 | 34.15% | -0.2996 | 0.432 | -15.066 | -13.655 | 68.06% | 1.23% | 511.30x | [-30.430, -8.574] | 0.994 | 0.025 |

## Per-regime PnL attribution

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 1 | 1.5085 | 0.7502 |
| TRANSITION | 58 | -21.8816 | -11.2999 |
| TREND | 64 | -45.9746 | -26.3013 |

## Per-strategy × per-regime attribution

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| flow_continuation | TREND | 64 | 21.88% | 0.332 | -45.9746 | -26.3013 |
| spring_reversal | RANGE | 1 | 100.00% | 0.000 | 1.5085 | 0.7502 |
| spring_reversal | TRANSITION | 58 | 46.55% | 0.544 | -21.8816 | -11.2999 |

## Tick-sample sanity check

The desk flow archive starts 2026-09-21, so no same-day archive/tick overlap exists. This validates the pipeline against direct ground truth on the tick day, not cross-source agreement.

Source: `/Volumes/WD My Passport 2626 Media/orderflow_lake/demo/data/tick_data_live/binance_futures/BTC_USDT/2026-08-05.parquet`; 1747079 ticks.

| Metric | Pipeline | Direct tick ground truth | Abs error | Result |
|---|---:|---:|---:|---|
| final_cvd | 3465.45400000 | 3465.45400000 | 0.0000000012 | PASS |
| last_15m_delta | -169.56400000 | -169.56400000 | 0.0000000000 | PASS |
| last_1h_delta | 67.52300000 | 67.52300000 | 0.0000000000 | PASS |
| last_6h_delta | 3091.45400000 | 3091.45400000 | 0.0000000000 | PASS |

## Flow-001 vs baseline-003

| Metric | Baseline-003 | Flow-001 | Delta |
|---|---:|---:|---:|
| trades | 121.0000 | 123.0000 | +2.0000 |
| win_rate | 0.1653 | 0.3415 | +0.1762 |
| expectancy_r | -0.3178 | -0.2996 | +0.0182 |
| profit_factor | 0.4584 | 0.4324 | -0.0260 |
| max_drawdown | 0.7052 | 0.6806 | -0.0246 |
| sharpe_daily | -22.8146 | -15.0660 | +7.7486 |
| turnover | 717.1778 | 511.2982 | -205.8796 |

## Disaster-session safety replays

| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 | 5 | -3.151 | 0 | 0 | 0 | FAIL |
| 2026-10-16 | 3 | -1.177 | 0 | 0 | 0 | PASS |
| 2026-10-22 | 3 | -0.278 | 0 | 0 | 0 | PASS |
| 2026-10-27 | 3 | -2.301 | 0 | 0 | 0 | PASS |
| 2026-10-29 | 0 | 0.000 | 0 | 0 | 0 | PASS |

## Gate verdict

- Flow track: FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': False, 'expectancy_r': False, 'churn_free': True}
- Disaster scenarios: FAIL
- Tick sanity: PASS
- **Overall: FAIL**

Flow-001 fails the preregistered gate. Diagnose the per-strategy/per-regime losses and STOP; P2b does not proceed.

## Data-power caveat

Only 30 days × 48 symbols of desk flow history exists. Statistical power comes from cross-symbol aggregation and a future forward shadow run, not from pretending there is more history. Tick samples validate feature semantics only and are not a performance track. A failed gate blocks P2b; this report is the honest P2a gate record.

# Baseline 002 — P1 regime-routed ensemble

## Headline out-of-sample results

Three expanding folds; thresholds were selected on train Sharpe only. Costs are 12 bps round-trip fees plus 5 bps slippage per fill. PBO is a fixed-seed daily-return sign-permutation estimate; DSR uses nine train trials.

| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC 2019-2026 5m | 1718 | 9.49% | -0.4828 | 0.040 | -10.048 | -8.923 | 298.84% | 1.33% | 2575.13x | [-11.170, -9.020] | 1.000 | 0.000 |
| Desk 30d flow (48 symbols) | 153 | 20.92% | -0.3655 | 0.227 | -117.748 | -18.874 | 93.82% | 0.20% | 823.08x | [-147.690, -104.957] | 1.000 | 0.000 |

## Per-regime PnL attribution

### BTC 2019-2026 5m

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 99 | -18.6725 | -60.2837 |
| TREND | 1619 | -280.1695 | -769.1677 |

### Desk 30d flow (48 symbols)

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 9 | -9.2637 | -5.9396 |
| TREND | 144 | -85.3497 | -49.9799 |

## Per-strategy × per-regime attribution

### BTC 2019-2026 5m

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| band_rejection | TREND | 967 | 8.17% | 0.037 | -164.9655 | -455.9680 |
| mean_reversion | RANGE | 99 | 1.01% | 0.000 | -18.6725 | -60.2837 |
| momentum | TREND | 652 | 12.73% | 0.051 | -115.2041 | -313.1997 |

Momentum alone is toxic on this track: 652 trades, 12.73% WR, 0.051 PF, -115.2041 net PnL.

### Desk 30d flow (48 symbols)

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| band_rejection | TREND | 76 | 25.00% | 0.374 | -31.9342 | -20.0516 |
| mean_reversion | RANGE | 9 | 0.00% | 0.000 | -9.2637 | -5.9396 |
| momentum | TREND | 68 | 19.12% | 0.139 | -53.4155 | -29.9283 |

Momentum alone is toxic on this track: 68 trades, 19.12% WR, 0.139 PF, -53.4155 net PnL.

## Baseline-001 vs baseline-002

| Track | Metric | Baseline-001 | Baseline-002 | Delta |
|---|---|---:|---:|---:|
| BTC 2019-2026 5m | trades | 1538.0000 | 1718.0000 | +180.0000 |
| BTC 2019-2026 5m | win_rate | 0.1372 | 0.0949 | -0.0423 |
| BTC 2019-2026 5m | profit_factor | 0.4022 | 0.0400 | -0.3622 |
| BTC 2019-2026 5m | max_drawdown | 2.9691 | 2.9884 | +0.0193 |
| BTC 2019-2026 5m | sharpe_daily | -4.0631 | -10.0484 | -5.9854 |
| BTC 2019-2026 5m | turnover | 2582.4282 | 2575.1257 | -7.3026 |
| Desk 30d flow (48 symbols) | trades | 62.0000 | 153.0000 | +91.0000 |
| Desk 30d flow (48 symbols) | win_rate | 0.0806 | 0.2092 | +0.1285 |
| Desk 30d flow (48 symbols) | profit_factor | 0.1818 | 0.2268 | +0.0450 |
| Desk 30d flow (48 symbols) | max_drawdown | 0.7614 | 0.9382 | +0.1768 |
| Desk 30d flow (48 symbols) | sharpe_daily | -27.0063 | -117.7482 | -90.7419 |
| Desk 30d flow (48 symbols) | turnover | 356.1467 | 823.0810 | +466.9343 |

| Track | Strategy | Trades 001 | Trades 002 | Net PnL 001 | Net PnL 002 | Delta PnL |
|---|---|---:|---:|---:|---:|---:|
| BTC 2019-2026 5m | band_rejection | 862 | 967 | -145.7706 | -164.9655 | -19.1948 |
| BTC 2019-2026 5m | momentum | 581 | 652 | -129.4832 | -115.2041 | +14.2791 |
| BTC 2019-2026 5m | mean_reversion | 95 | 99 | -21.5737 | -18.6725 | +2.9012 |
| Desk 30d flow (48 symbols) | band_rejection | 28 | 76 | -26.1467 | -31.9342 | -5.7874 |
| Desk 30d flow (48 symbols) | momentum | 29 | 68 | -46.6509 | -53.4155 | -6.7646 |
| Desk 30d flow (48 symbols) | mean_reversion | 5 | 9 | -3.8074 | -9.2637 | -5.4563 |

## Disaster-session safety replays

| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 | 6 | -1.086 | 0 | 0 | 0 | PASS |
| 2026-10-16 | 3 | -0.849 | 0 | 0 | 0 | PASS |
| 2026-10-22 | 6 | -2.530 | 0 | 0 | 0 | PASS |
| 2026-10-27 | 4 | -2.689 | 0 | 0 | 0 | PASS |
| 2026-10-29 | 3 | -1.523 | 0 | 0 | 0 | PASS |

## Gate verdict

- BTC 2019-2026 5m: FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': True, 'churn_free': True}
- Desk 30d flow (48 symbols): FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': True, 'churn_free': True}
- Disaster scenarios: PASS
- **Overall: FAIL**

The single structural exit iteration still fails. Stop here: no further parameter tuning, alternate exit mode, or second structural iteration is permitted under this plan.

## Limitations

Stops use bar touches with stop-first ambiguity, targets use the target price plus adverse slippage, and there is no funding, liquidation, queue, or partial-fill model. BTC has no flow events, so its flow veto is inactive. The optional LightGBM baseline was skipped because its dependency download timed out; it did not block this deterministic baseline. These are backtest results, not evidence of live profitability.

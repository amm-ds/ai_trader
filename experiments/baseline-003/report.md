# Baseline 003 — P1 regime-routed ensemble

## Headline out-of-sample results

Three expanding folds; thresholds were selected on train Sharpe only. Costs are 12 bps round-trip fees plus 5 bps slippage per fill. PBO is a fixed-seed daily-return sign-permutation estimate; DSR uses nine train trials.

| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC 2019-2026 5m | 1390 | 14.39% | -0.4427 | 0.308 | -5.647 | -6.900 | 284.78% | 8.82% | 2697.29x | [-7.298, -4.284] | 1.000 | 0.000 |
| Desk 30d flow (48 symbols) | 121 | 16.53% | -0.3178 | 0.458 | -22.815 | -15.473 | 70.52% | 1.20% | 717.18x | [-48.144, -13.946] | 1.000 | 0.010 |

Structural finding: momentum was removed after baseline-001 toxicity (13.43% WR / 0.324 PF on BTC; 3.45% WR / 0.015 PF on flow). Baseline-003 routes TREND to band_rejection only and keeps mean_reversion RANGE-only.

## Per-regime PnL attribution

### BTC 2019-2026 5m

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 94 | -25.1652 | -61.4616 |
| TREND | 1296 | -270.3267 | -553.8786 |

### Desk 30d flow (48 symbols)

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 7 | -6.0712 | -3.5570 |
| TREND | 114 | -62.9360 | -34.9004 |

## Per-strategy × per-regime attribution

### BTC 2019-2026 5m

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| band_rejection | TREND | 1296 | 15.35% | 0.328 | -270.3267 | -553.8786 |
| mean_reversion | RANGE | 94 | 1.06% | 0.000 | -25.1652 | -61.4616 |

### Desk 30d flow (48 symbols)

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| band_rejection | TREND | 114 | 17.54% | 0.481 | -62.9360 | -34.9004 |
| mean_reversion | RANGE | 7 | 0.00% | 0.000 | -6.0712 | -3.5570 |

## Baseline-001 vs baseline-002 vs baseline-003

| Track | Metric | Baseline-001 | Baseline-002 | Baseline-003 | Delta 003-001 |
|---|---|---:|---:|---:|---:|
| BTC 2019-2026 5m | trades | 1538.0000 | 1718.0000 | 1390.0000 | -148.0000 |
| BTC 2019-2026 5m | win_rate | 0.1372 | 0.0949 | 0.1439 | +0.0067 |
| BTC 2019-2026 5m | profit_factor | 0.4022 | 0.0400 | 0.3084 | -0.0938 |
| BTC 2019-2026 5m | max_drawdown | 2.9691 | 2.9884 | 2.8478 | -0.1212 |
| BTC 2019-2026 5m | sharpe_daily | -4.0631 | -10.0484 | -5.6471 | -1.5840 |
| BTC 2019-2026 5m | turnover | 2582.4282 | 2575.1257 | 2697.2885 | +114.8602 |
| Desk 30d flow (48 symbols) | trades | 62.0000 | 153.0000 | 121.0000 | +59.0000 |
| Desk 30d flow (48 symbols) | win_rate | 0.0806 | 0.2092 | 0.1653 | +0.0846 |
| Desk 30d flow (48 symbols) | profit_factor | 0.1818 | 0.2268 | 0.4584 | +0.2766 |
| Desk 30d flow (48 symbols) | max_drawdown | 0.7614 | 0.9382 | 0.7052 | -0.0562 |
| Desk 30d flow (48 symbols) | sharpe_daily | -27.0063 | -117.7482 | -22.8146 | +4.1917 |
| Desk 30d flow (48 symbols) | turnover | 356.1467 | 823.0810 | 717.1778 | +361.0311 |

| Track | Strategy | Trades 001 | Trades 002 | Trades 003 | Net PnL 001 | Net PnL 002 | Net PnL 003 | Delta 003-001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTC 2019-2026 5m | band_rejection | 862 | 967 | 1296 | -145.7706 | -164.9655 | -270.3267 | -124.5560 |
| BTC 2019-2026 5m | momentum | 581 | 652 | 0 | -129.4832 | -115.2041 | 0.0000 | +129.4832 |
| BTC 2019-2026 5m | mean_reversion | 95 | 99 | 94 | -21.5737 | -18.6725 | -25.1652 | -3.5914 |
| Desk 30d flow (48 symbols) | band_rejection | 28 | 76 | 114 | -26.1467 | -31.9342 | -62.9360 | -36.7892 |
| Desk 30d flow (48 symbols) | momentum | 29 | 68 | 0 | -46.6509 | -53.4155 | 0.0000 | +46.6509 |
| Desk 30d flow (48 symbols) | mean_reversion | 5 | 9 | 7 | -3.8074 | -9.2637 | -6.0712 | -2.2639 |

## Disaster-session safety replays

| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 | 2 | -0.706 | 0 | 0 | 0 | PASS |
| 2026-10-16 | 3 | 1.261 | 0 | 0 | 0 | PASS |
| 2026-10-22 | 5 | -0.800 | 0 | 0 | 0 | PASS |
| 2026-10-27 | 3 | -2.361 | 0 | 0 | 0 | PASS |
| 2026-10-29 | 2 | -2.000 | 0 | 0 | 0 | PASS |

## Gate verdict

- BTC 2019-2026 5m: FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': True, 'churn_free': True, 'turnover_target': False}
- Desk 30d flow (48 symbols): FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': True, 'churn_free': True}
- Disaster scenarios: PASS
- **Overall: FAIL**

Baseline-003 fails and is the final P1 gate record. Momentum removal is retained as an evidence-based finding, but 15m band-only swing execution does not pass. STOP: no further structural iterations or parameter tuning.

## Limitations

Stops use bar touches with stop-first ambiguity, targets use the target price plus adverse slippage, and there is no funding, liquidation, queue, or partial-fill model. BTC has no flow events, so its flow veto is inactive. The optional LightGBM baseline was skipped because its dependency download timed out; it did not block this deterministic baseline. These are backtest results, not evidence of live profitability.

# Baseline 001 — P1 regime-routed ensemble

## Headline out-of-sample results

Three expanding folds; thresholds were selected on train Sharpe only. Costs are 12 bps round-trip fees plus 5 bps slippage per fill. PBO is a fixed-seed daily-return sign-permutation estimate; DSR uses nine train trials.

| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC 2019-2026 5m | 1538 | 13.72% | -0.4201 | 0.402 | -4.063 | -6.010 | 296.91% | 15.79% | 2582.43x | [-6.096, -2.585] | 1.000 | 0.000 |
| Desk 30d flow (48 symbols) | 62 | 8.06% | -0.6985 | 0.182 | -27.006 | -16.455 | 76.14% | 1.24% | 356.15x | [-84.166, -16.976] | 1.000 | 0.025 |

## Per-regime PnL attribution

### BTC 2019-2026 5m

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 95 | -21.5737 | -58.4974 |
| TREND | 1443 | -275.2538 | -587.6480 |

### Desk 30d flow (48 symbols)

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| RANGE | 5 | -3.8074 | -2.3727 |
| TREND | 57 | -72.7976 | -40.9362 |

## Per-strategy × per-regime attribution

### BTC 2019-2026 5m

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| band_rejection | TREND | 862 | 15.31% | 0.486 | -145.7706 | -338.5455 |
| mean_reversion | RANGE | 95 | 1.05% | 0.001 | -21.5737 | -58.4974 |
| momentum | TREND | 581 | 13.43% | 0.324 | -129.4832 | -249.1025 |

Momentum alone is toxic on this track: 581 trades, 13.43% WR, 0.324 PF, -129.4832 net PnL.

### Desk 30d flow (48 symbols)

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| band_rejection | TREND | 28 | 14.29% | 0.384 | -26.1467 | -14.6874 |
| mean_reversion | RANGE | 5 | 0.00% | 0.000 | -3.8074 | -2.3727 |
| momentum | TREND | 29 | 3.45% | 0.015 | -46.6509 | -26.2488 |

Momentum alone is toxic on this track: 29 trades, 3.45% WR, 0.015 PF, -46.6509 net PnL.

## Disaster-session safety replays

| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 | 4 | 10.406 | 0 | 0 | 0 | PASS |
| 2026-10-16 | 3 | -2.023 | 0 | 0 | 0 | PASS |
| 2026-10-22 | 2 | 5.338 | 0 | 0 | 0 | PASS |
| 2026-10-27 | 3 | -3.000 | 0 | 0 | 0 | PASS |
| 2026-10-29 | 2 | -2.000 | 0 | 0 | 0 | PASS |

## Gate verdict

- BTC 2019-2026 5m: FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': True, 'churn_free': True}
- Desk 30d flow (48 symbols): FAIL — {'profit_factor': False, 'max_drawdown': False, 'trade_count': True, 'churn_free': True}
- Disaster scenarios: PASS
- **Overall: FAIL**

The ensemble does not pass. Failed track(s): BTC 2019-2026 5m, Desk 30d flow (48 symbols). TREND attribution dominates losses on both tracks and turnover is extreme. Next concrete fix: split TREND attribution by band-rejection versus momentum and repair the stop-only trend exit with a structural break/target exit, evaluated independently before any AI or further threshold work. Do not run a second grid iteration.

## Limitations

Stops use bar touches with stop-first ambiguity, targets use the target price plus adverse slippage, and there is no funding, liquidation, queue, or partial-fill model. BTC has no flow events, so its flow veto is inactive. The optional LightGBM baseline was skipped because its dependency download timed out; it did not block this deterministic baseline. These are backtest results, not evidence of live profitability.

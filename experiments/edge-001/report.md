# Edge-001 — deep BTC microstructure

Native LightGBM + three-fold purged OOF isotonic calibration, five expanding outer folds with one-day embargo. ETH/SOL materialization is pending; this is a complete BTC experiment.

## Headline out-of-sample result

| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC deep ticks 2019-2026 | 0 | 0.00% | 0.0000 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | 0.00x | [0.000, 0.000] | 1.000 | 0.000 |

## Per-regime attribution

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|

## Per-strategy × per-regime attribution

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|

## Model diagnostics

| Fold | Flat band | Threshold | Quantile | Flat-excluded rows | Accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0015 | 0.4232 | 0.60 | 16886 | 54.87% | 0.5642 |
| 2 | 0.0010 | 0.4473 | 0.60 | 20199 | 54.01% | 0.5580 |
| 3 | 0.0015 | 0.4083 | 0.60 | 19423 | 53.15% | 0.5450 |
| 4 | 0.0010 | 0.4386 | 0.60 | 21348 | 51.67% | 0.5257 |
| 5 | 0.0015 | 0.3839 | 0.60 | 17858 | 52.81% | 0.5377 |

### Top-10 gain importances

| Fold | Rank | Feature | Gain share |
|---|---:|---|---:|
| 1 | 1 | atr_pct | 0.3397 |
| 1 | 2 | sweep_sell | 0.2484 |
| 1 | 3 | sweep_buy | 0.1144 |
| 1 | 4 | distance_ema9 | 0.0484 |
| 1 | 5 | distance_bb_mid | 0.0414 |
| 1 | 6 | microprice_deviation_ewma | 0.0315 |
| 1 | 7 | return_15m | 0.0312 |
| 1 | 8 | dollar_volume_z | 0.0282 |
| 1 | 9 | rsi3 | 0.0238 |
| 1 | 10 | return_1h | 0.0226 |
| 2 | 1 | atr_pct | 0.4951 |
| 2 | 2 | sweep_sell | 0.1250 |
| 2 | 3 | sweep_buy | 0.1234 |
| 2 | 4 | distance_ema9 | 0.0632 |
| 2 | 5 | return_15m | 0.0327 |
| 2 | 6 | distance_bb_mid | 0.0318 |
| 2 | 7 | microprice_deviation_ewma | 0.0227 |
| 2 | 8 | cvd_slope_1h | 0.0201 |
| 2 | 9 | rsi3 | 0.0196 |
| 2 | 10 | dollar_volume_z | 0.0181 |
| 3 | 1 | atr_pct | 0.5550 |
| 3 | 2 | sweep_buy | 0.1252 |
| 3 | 3 | sweep_sell | 0.1153 |
| 3 | 4 | distance_ema9 | 0.0469 |
| 3 | 5 | return_15m | 0.0313 |
| 3 | 6 | distance_bb_mid | 0.0235 |
| 3 | 7 | microprice_deviation_ewma | 0.0229 |
| 3 | 8 | cvd_slope_1h | 0.0154 |
| 3 | 9 | rsi3 | 0.0142 |
| 3 | 10 | dollar_volume_z | 0.0137 |
| 4 | 1 | atr_pct | 0.5341 |
| 4 | 2 | sweep_buy | 0.1324 |
| 4 | 3 | sweep_sell | 0.1250 |
| 4 | 4 | distance_ema9 | 0.0587 |
| 4 | 5 | return_15m | 0.0287 |
| 4 | 6 | distance_bb_mid | 0.0211 |
| 4 | 7 | microprice_deviation_ewma | 0.0202 |
| 4 | 8 | rsi3 | 0.0157 |
| 4 | 9 | dollar_volume_z | 0.0133 |
| 4 | 10 | cvd_slope_1h | 0.0119 |
| 5 | 1 | atr_pct | 0.5479 |
| 5 | 2 | sweep_buy | 0.1445 |
| 5 | 3 | sweep_sell | 0.1312 |
| 5 | 4 | distance_ema9 | 0.0402 |
| 5 | 5 | return_15m | 0.0296 |
| 5 | 6 | microprice_deviation_ewma | 0.0210 |
| 5 | 7 | distance_bb_mid | 0.0181 |
| 5 | 8 | rsi3 | 0.0127 |
| 5 | 9 | cvd_slope_1h | 0.0120 |
| 5 | 10 | dollar_volume_z | 0.0100 |

### Calibration summary

| Fold | Up-probability bin | Count | Mean probability | Empirical up rate |
|---|---|---:|---:|---:|
| 1 | 0.0-0.2 | 0 | 0.0000 | 0.0000 |
| 1 | 0.2-0.4 | 427 | 0.3702 | 0.4426 |
| 1 | 0.4-0.6 | 16000 | 0.4963 | 0.5064 |
| 1 | 0.6-0.8 | 448 | 0.6243 | 0.5670 |
| 1 | 0.8-1.0 | 11 | 1.0000 | 0.3636 |
| 2 | 0.0-0.2 | 118 | 0.0000 | 0.5085 |
| 2 | 0.2-0.4 | 465 | 0.3784 | 0.4452 |
| 2 | 0.4-0.6 | 19257 | 0.4899 | 0.5191 |
| 2 | 0.6-0.8 | 329 | 0.6373 | 0.5927 |
| 2 | 0.8-1.0 | 30 | 0.9952 | 0.6000 |
| 3 | 0.0-0.2 | 2 | 0.0000 | 0.5000 |
| 3 | 0.2-0.4 | 528 | 0.3834 | 0.4015 |
| 3 | 0.4-0.6 | 18016 | 0.4942 | 0.5121 |
| 3 | 0.6-0.8 | 876 | 0.6306 | 0.5445 |
| 3 | 0.8-1.0 | 1 | 0.8187 | 0.0000 |
| 4 | 0.0-0.2 | 65 | 0.0013 | 0.6462 |
| 4 | 0.2-0.4 | 135 | 0.3800 | 0.4815 |
| 4 | 0.4-0.6 | 20833 | 0.5006 | 0.5065 |
| 4 | 0.6-0.8 | 315 | 0.6236 | 0.6317 |
| 4 | 0.8-1.0 | 0 | 0.0000 | 0.0000 |
| 5 | 0.0-0.2 | 0 | 0.0000 | 0.0000 |
| 5 | 0.2-0.4 | 219 | 0.3837 | 0.5114 |
| 5 | 0.4-0.6 | 16866 | 0.5059 | 0.4948 |
| 5 | 0.6-0.8 | 773 | 0.6289 | 0.5459 |
| 5 | 0.8-1.0 | 0 | 0.0000 | 0.0000 |

## Baseline Sharpe comparison

| System | Daily Sharpe |
|---|---:|
| buy_hold | 0.8632 |
| ma_cross | -0.6315 |
| quant_only_ofi | -1.7335 |
| edge-001 | 0.0000 |
| baseline-003 band system | -5.6471 |

## Stress windows

| Window | Trades | PF | Max DD | Sharpe |
|---|---:|---:|---:|---:|
| 2021-bear | 1 | 0.000 | 0.73% | -2.003 |
| 2024-stress | 0 | 0.000 | 0.00% | 0.000 |

## Disaster-session safety replays (BTC)

| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-16 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-22 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-27 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-29 | 0 | 0.000 | 0 | 0 | 0 | PASS |

## E0 materialization and tick sanity

BTC files: 3683; chunks: 3683; ticks: 3630031122; feature rows: 241871; throughput: 1.55 files/s, 1527401 ticks/s.

| Check | Absolute error | Result |
|---|---:|---|
| CVD | 0.0000000004 | PASS |
| Last 15m delta | 0.0000000000 | PASS |
| Imbalance | 0.0000000000 | PASS |

## Gate verdict

- Edge track: FAIL — {'profit_factor': False, 'max_drawdown': True, 'trade_count': False, 'expectancy_r': False, 'beats_buy_hold_sharpe': False, 'churn_free': True}
- Disaster scenarios: PASS
- Tick sanity: PASS
- **Overall: FAIL**

Edge-001 fails. Deep learned microstructure does not pass the gate. STOP: the AI layer is not authorized.

Ten years of tick history does not guarantee live edge. OOF calibration and deep samples do not create independent future regimes; forward shadow/paper validation remains the arbiter.

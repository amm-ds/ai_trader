# Flow-002 — learned flow-features model

Backend: **native LightGBM 4.7.0 Booster** (installed on the longer-timeout retry; its sklearn wrapper was unavailable because sklearn is absent, so the native API is used with balanced sample weights).

## Headline out-of-sample result

| Track | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Exposure | Turnover | Sharpe 95% CI | PBO | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Desk 30d flow (48 symbols) | 3 | 0.00% | -1.0000 | 0.000 | -9.228 | -8.544 | 5.96% | 0.00% | 22.49x | [-15.069, 0.000] | 1.000 | 0.003 |

## Per-regime PnL attribution

| Regime | Trades | Net PnL | Sum R |
|---|---:|---:|---:|
| TRANSITION | 2 | -3.9600 | -2.0000 |
| TREND | 1 | -2.0000 | -1.0000 |

## Per-strategy × per-regime attribution

| Strategy | Regime | Trades | WR | PF | Net PnL | Sum R |
|---|---|---:|---:|---:|---:|---:|
| learned_flow | TRANSITION | 2 | 0.00% | 0.000 | -3.9600 | -2.0000 |
| learned_flow | TREND | 1 | 0.00% | 0.000 | -2.0000 | -1.0000 |

## Model diagnostics

| Fold | Threshold | Flat-excluded rows | Accuracy | AUC |
|---|---:|---:|---:|---:|
| 1 | 0.60 | 19342 | 52.50% | 0.5285 |
| 2 | 0.60 | 19177 | 52.95% | 0.5280 |
| 3 | 0.60 | 19764 | 49.84% | 0.4993 |

### Top-10 gain importances by fold

| Fold | Rank | Feature | Gain share |
|---|---:|---|---:|
| 1 | 1 | atr_pct | 0.5691 |
| 1 | 2 | cvd_slope_pct | 0.0808 |
| 1 | 3 | bb_width_pct | 0.0421 |
| 1 | 4 | realized_vol_pct | 0.0414 |
| 1 | 5 | return_1h | 0.0398 |
| 1 | 6 | distance_ema9 | 0.0241 |
| 1 | 7 | rsi3 | 0.0174 |
| 1 | 8 | distance_bb_mid | 0.0172 |
| 1 | 9 | trend_4h | 0.0159 |
| 1 | 10 | cls5_cls6_count_ratio_6h | 0.0139 |
| 2 | 1 | atr_pct | 0.6064 |
| 2 | 2 | cvd_slope_pct | 0.0539 |
| 2 | 3 | realized_vol_pct | 0.0475 |
| 2 | 4 | bb_width_pct | 0.0425 |
| 2 | 5 | return_1h | 0.0416 |
| 2 | 6 | distance_ema9 | 0.0265 |
| 2 | 7 | return_15m | 0.0173 |
| 2 | 8 | distance_bb_mid | 0.0161 |
| 2 | 9 | rsi3 | 0.0154 |
| 2 | 10 | trend_1h | 0.0131 |
| 3 | 1 | atr_pct | 0.6377 |
| 3 | 2 | realized_vol_pct | 0.0501 |
| 3 | 3 | bb_width_pct | 0.0469 |
| 3 | 4 | return_1h | 0.0354 |
| 3 | 5 | cvd_slope_pct | 0.0329 |
| 3 | 6 | distance_ema9 | 0.0227 |
| 3 | 7 | trend_4h | 0.0157 |
| 3 | 8 | distance_bb_mid | 0.0150 |
| 3 | 9 | rsi3 | 0.0139 |
| 3 | 10 | return_15m | 0.0138 |

### Probability calibration summary

| Fold | Up-probability bin | Count | Mean probability | Empirical up rate |
|---|---|---:|---:|---:|
| 1 | 0.0-0.2 | 0 | 0.0000 | 0.0000 |
| 1 | 0.2-0.4 | 763 | 0.3762 | 0.5216 |
| 1 | 0.4-0.6 | 17439 | 0.4905 | 0.4761 |
| 1 | 0.6-0.8 | 1133 | 0.6535 | 0.5825 |
| 1 | 0.8-1.0 | 7 | 0.8084 | 0.5714 |
| 2 | 0.0-0.2 | 0 | 0.0000 | 0.0000 |
| 2 | 0.2-0.4 | 371 | 0.3792 | 0.3693 |
| 2 | 0.4-0.6 | 17786 | 0.4882 | 0.4343 |
| 2 | 0.6-0.8 | 1019 | 0.6468 | 0.5868 |
| 2 | 0.8-1.0 | 1 | 0.8192 | 1.0000 |
| 3 | 0.0-0.2 | 0 | 0.0000 | 0.0000 |
| 3 | 0.2-0.4 | 170 | 0.3786 | 0.4882 |
| 3 | 0.4-0.6 | 16409 | 0.4980 | 0.4966 |
| 3 | 0.6-0.8 | 3145 | 0.6692 | 0.4747 |
| 3 | 0.8-1.0 | 40 | 0.8125 | 0.6250 |

## Flow-002 vs flow-001 and baseline-003

| Metric | Baseline-003 | Flow-001 | Flow-002 | Delta 002-001 |
|---|---:|---:|---:|---:|
| trades | 121.0000 | 123.0000 | 3.0000 | -120.0000 |
| win_rate | 0.1653 | 0.3415 | 0.0000 | -0.3415 |
| expectancy_r | -0.3178 | -0.2996 | -1.0000 | -0.7004 |
| profit_factor | 0.4584 | 0.4324 | 0.0000 | -0.4324 |
| max_drawdown | 0.7052 | 0.6806 | 0.0596 | -0.6210 |
| sharpe_daily | -22.8146 | -15.0660 | -9.2281 | +5.8380 |
| turnover | 717.1778 | 511.2982 | 22.4937 | -488.8045 |

## Disaster-session safety replays

| Session | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 | 4 | -1.893 | 0 | 0 | 0 | PASS |
| 2026-10-16 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-22 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-27 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-29 | 0 | 0.000 | 0 | 0 | 0 | PASS |

## Gate verdict

- Flow track: FAIL — {'profit_factor': False, 'max_drawdown': True, 'trade_count': False, 'expectancy_r': False, 'churn_free': True}
- Disaster scenarios: PASS
- **Overall: FAIL**

Flow-002 fails. The learned model does not repair the flow baseline. STOP: this is the final experiment in the line.

## Data-power caveat

Only 30 days × 48 symbols of flow history exists. Pooling symbols adds samples but not independent market regimes. Walk-forward tests, flat-excluded diagnostics, and calibration do not establish live edge. Flow-002 is the final experiment in this line; a failed gate stops further model iteration.

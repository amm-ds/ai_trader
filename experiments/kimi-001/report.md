# Kimi-001 — does Kimi improve trading metrics net of all costs?

Model: `kimi-for-coding` via the OpenAI-compatible Moonshot API, prompt `kimi-001-prompt-v1`.
Key material stays in the gitignored repo `.env`; the client enforces one concurrent call,
exponential backoff on 429/5xx, and a 600-call-per-UTC-day cap. Every decision is cached by
(snapshot_hash, model, prompt_version), so replays never re-call the API.

## Token economics (ESTIMATE prices — verify at platform.moonshot.ai)

| Metric | Value |
|---|---:|
| LLM calls (cached + live) | 978 |
| Prompt tokens | 1100426 |
| Completion tokens | 832935 |
| Total tokens | 1933361 |
| Token USD | $2.7426 |
| Invalid-JSON rate | 0.00% |
| Latency mean / median / P95 (s) | 20.79 / 18.76 / 42.48 |

## Arm comparison (backtest OOS snapshot dataset, net of fees + slippage + tokens)

| Arm | Trades | WR | E[R] | PF | Sharpe | Sortino | Max DD | Turnover | Fees $ | Slippage $ | Token $ | Net after everything $ | Cost per R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A quant-only | 0 | 0.00% | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.00% | 0.00x | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B LLM-proposed | 11 | 18.18% | -0.5594 | 0.163 | -1.0658 | -1.1483 | 12.59% | 63.53x | 3.8116 | 3.1748 | 2.7326 | -15.0387 | 0.441775 |
| C hybrid reviewer | 0 | 0.00% | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.00% | 0.00x | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Buy-and-hold daily Sharpe over the same OOS folds: 0.8632.
Tokens per arm-B decision (base prompt): 4812.7.

### Arm B proposal accounting

| Outcome | Count |
|---|---:|
| long | 5 |
| short | 6 |
| veto:evening-bell | 2 |
| veto:llm-flat | 387 |

## Session and stress replays (arm B active)

| Window | Kind | LLM reviews | Trades | Min day R | Counter-trend | Cooldown | Pre-cascade | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-10-08 | session | 6 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-16 | session | 6 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-22 | session | 6 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-27 | session | 6 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2026-10-29 | session | 6 | 0 | 0.000 | 0 | 0 | 0 | PASS |
| 2021-bear | stress | 273 | 16 | -1.000 | 0 | 0 | 0 | PASS |
| 2024-stress | stress | 270 | 12 | -1.000 | 0 | 0 | 0 | PASS |

## Prompt-sensitivity spot check

5/5 re-asked snapshots (reordered context) returned the same action.

## Verdict (preregistered rule)

Rule: B beats A and buy-and-hold on net-after-everything Sharpe, PF>1.3, maxDD<20% on backtest OOS, and no behavioral breach in replays.

**NO** — arm B net-after-everything Sharpe -1.0658 vs A 0.0000 and buy-and-hold 0.8632; PF 0.163 (>1.3 required); max DD 12.59% (<20% required); replays behavioral PASS.

The governor stayed sovereign throughout: every arm-B order passed the unchanged `Governor.review`,
and non-compliant proposals were vetoed with logged reasons. Token prices are estimate-marked.

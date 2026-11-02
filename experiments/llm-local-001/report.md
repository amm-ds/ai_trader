# LLM-local-001 — local decision benchmark

Requested `qwen3.8:27b-mlx`; its 18 GB pull failed with `no space left on device`. Benchmark used **`qwen3.6:35b-a3b-q4_K_M`**, already installed. Ollama was already serving on localhost:11434.

## Frozen-snapshot benchmark

| Model | Snapshots | Agreement | Invalid JSON | Mean latency (s) | Median (s) | P95 (s) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3.6:35b-a3b-q4_K_M | 60 | 96.67% | 0.00% | 56.753 | 53.299 | 87.306 | 41.554 |

## Counterfactual OOS replay

| Arm | Trades | Sharpe | PF | Max DD | E[R] |
|---|---:|---:|---:|---:|---:|
| Quant-only | 0 | 0.0000 | 0.0000 | 0.00% | 0.0000 |
| Quant + local LLM | 0 | 0.0000 | 0.0000 | 0.00% | 0.0000 |

## Verdict

**NO MEASURABLE VALUE** The LLM arm can only veto or halve existing entries; it cannot add trades or increase size. Edge-001 produced zero quant entries, so there is no measurable counterfactual improvement to claim. Latency/agreement diagnostics are reported, not marketed as trading edge.

The read-only v11 assist review is at `docs/trading_bots_crypto_ai_review.md`.

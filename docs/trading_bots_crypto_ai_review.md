# Read-only AI-assist review — trading_bots_crypto v11

## Scope and evidence

Reviewed read-only: `AGENTS.md`, `nautilus_btc/strategies/v11_edge.py`,
`scripts/refresh_v11_features.py`, `results/v11_edge_staleness_experiment.md`,
`results/v11_meta_stack_backtest.md`, and recent `paper_v11_edge` / refresh logs.
No files in `trading_bots_crypto` were changed and no launchd agent was touched.

The validated core is the causal feature pipeline, model/calibration artifacts,
cost and meta gates, and the paper-node signal/order path. Those are not proposal
targets. The appropriate role for a local LLM is explanation and triage over
read-only outputs, not trading authority.

## Top three recommendations

### 1. Blocked/taken-trade triage and explanation

- **Touch:** read-only `results/paper_v11_edge.out/.err`, the exact close-labeled
  feature row, model confidence, cost-gate arithmetic, and existing veto reasons.
  Emit structured daily JSON/Markdown in `ai_trader` linking each explanation to
  its evidence row.
- **Never touch:** `V11EdgeConfig`, `V11EdgeStrategy`, model pickles, thresholds,
  pending-retry behavior, order intents, or the launchd paper node.
- **Expected value:** fastest way to distinguish fee-negative blocks, stale-data
  blocks, regime/meta vetoes, and actual entries. The staleness report shows why
  this context matters: one-bar lag materially changed outcomes.
- **M4 Max cost:** asynchronous batch of a few calls/day; no per-bar inference.
  Use a small local model first and cap evidence to the fields already logged.
- **Safe prototype:** **YES** — an `ai_trader` adapter over copied/read-only log
  and parquet outputs. No adapter writes back.

### 2. Kill-switch, refresh, and infrastructure anomaly narrative

- **Touch:** read-only refresh and paper logs. Current examples include repeated
  `No space left on device`, WebSocket pong-deadline expiry, and earlier fetch
  failures alongside successful 15-minute refresh completions. Produce incident
  summaries with first/last occurrence, count, affected subsystem, and links to
  the exact log lines.
- **Never touch:** kill-switch state, restart policy, launchd plists, credentials,
  exchange clients, or suppression of an alarm. The LLM cannot clear an incident.
- **Expected value:** converts repetitive logs into actionable operator review
  while preserving deterministic alarm ownership. Particularly useful for disk,
  freshness, and transport failures that can silently invalidate a paper run.
- **M4 Max cost:** one call per incident batch/day, not per repeated line.
  Deduplicate deterministic error signatures before inference.
- **Safe prototype:** **YES** — read-only incident adapter with human review.

### 3. Daily journal synthesis and regime-context text research

- **Touch:** daily trades/blocks/incidents plus the corresponding frozen feature
  rows. Produce an append-only daily journal: what was allowed, what was blocked,
  freshness health, and which evidence needs human review.
- **Never touch:** validated trade ledger semantics, fill IDs, timestamps, or the
  decision to promote a configuration. Do not let generated journal text become
  a hidden parameter change.
- **Expected value:** better auditability and faster daily review without changing
  trading behavior. As a separate research-only extension, regime narratives can
  be evaluated as text features for the meta-labeler **only** when generated
  point-in-time from already-available rows, with walk-forward ablations and an
  explicit prohibition on forward-looking summaries, post-trade outcomes, or
  same-bar journal text. No narrative feature belongs in the live meta-labeler
  without passing the existing purged protocol.
- **M4 Max cost:** one daily synthesis call; research embeddings/text features
  must be precomputed offline, never generated inside the 15-minute live path.
- **Safe prototype:** **YES** for the journal. **RESEARCH-ONLY** for text
  features, inside `ai_trader` with leakage tests and no production writes.

## Lower-priority but safe: alert summarization

Aggregate repetitive data-refresh, instrument, and transport alerts into a short
operator digest. Keep exact counts and raw links; never rank away a kill-switch
or freshness error. This is a read-only adapter and has low batch cost, but it
should follow, not precede, trade triage and anomaly narratives.

## Explicit non-opportunities

- LLM-generated entries, size increases, stop changes, or order-routing changes.
- Prompt-time use of future bars, later trade outcomes, or post-close journals as
  model features.
- Replacing cost/meta gates with natural-language judgment.
- Autonomous restarts, credential handling, or launchd administration.
- Calling the model on every 15-minute decision before measured batch evidence
  shows a benefit. The `llm-local-001` counterfactual provides no such evidence.

## Prototype protocol

1. Copy or read logs/parquets read-only into an `ai_trader` adapter.
2. Normalize to frozen evidence IDs and structured outputs.
3. Generate append-only reports outside `trading_bots_crypto`.
4. Evaluate factual accuracy against the raw rows before any operator use.
5. Keep the validated strategy, model, launchd, and order paths unchanged.

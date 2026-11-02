# ADR 0001: Three intelligences and sovereign risk

- Status: accepted
- Date: 2026-10-31

## Context

Trading systems can mix statistical predictions, semantic analysis, and order
execution. Allowing any probabilistic component to directly control exposure
makes failures difficult to bound and reproduce.

## Decision

Separate statistical intelligence (features/models), semantic intelligence
(future Kimi agents), and operational intelligence (deterministic risk and
execution). Only the governor may veto or resize an order; it may only shrink.
All components consume timestamped snapshots and emit auditable decisions.

## Consequences

Simulation and future live runners share the same strategy/risk/broker path.
Model improvements must be measured against a frozen quant baseline and cannot
waive safety limits. This adds integration discipline but makes failure modes
explicit and replayable.

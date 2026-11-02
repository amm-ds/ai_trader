# Sovereign risk engine

`Governor.review` is deterministic and side-effect free. Its context contains
all relevant state, and its decision includes machine-readable veto reasons.

Phase-0 limits: 1R = 2% of current equity; daily realized loss halt at -3R;
four-hour per-symbol cooldown; at most one re-entry per symbol/day with a written
trigger; three total positions; two positions per correlation group; no new risk
at or after 19:00Z; no same-symbol adding/averaging down; directional stop
binding and 0.1%-5% stop range; stale/future data rejection; configured kill
switch; and rejection when observed slippage exceeds twice the model.

Size is capped at `equity * 0.02 / estimated_stop_loss_per_unit`, where estimated
stop loss includes adverse entry/exit slippage and both fill fees. Approved size
is always the minimum of requested and permitted quantity, and a stop at the
modeled price is therefore -1R net. A veto is fail-closed. The backtester
flattens at the evening bell and does not hold overnight.

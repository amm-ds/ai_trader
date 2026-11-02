# Architecture

The research slice normalizes timestamped klines and flow events into immutable,
per-symbol views. Resampled bars are labeled by their right-exclusive close time.
Band, ATR, RSI, ROC, volatility, and percentile indicators use completed bars
only. The P1 ensemble routes band-rejection and momentum in TREND, mean-reversion
in RANGE, and stands down in VOLATILE or TRANSITION. Every routing decision keeps
its regime and strategy evidence.

Every proposed order passes through the deterministic governor. It can reject or
shrink, never expand. Approved orders go to the paper broker, which owns fills,
fees, stops, persistence, and R accounting. The event-driven backtester composes
these production-bound components rather than maintaining a second simulation
implementation. Baseline-002 applies one fixed trend-exit structure: prior-swing
or 1h-band targets, +1R breakeven arming, a never-widen 3×ATR chandelier, and
completed-5m EMA9 breaks with flow agreement. Mean-reversion exits are unchanged.
Baseline-003 removes momentum, executes on completed 15m bars, and uses only the
structural stop, +1R breakeven, and 15m 3×ATR chandelier for swing trend exits.
P2a pivots to flow-first entries: bands supply regime/structure only, cls5/cls6
intensity and CVD confirm 15m spring reclaims or with-trend continuation resumes.
Flow-002 replaces hand thresholds with a native LightGBM multiclass Booster over
causal flow, band, and price context; probabilities still pass the 15m structure
gate and sovereign governor before any entry.

P1 evaluates the deterministic ensemble with three expanding walk-forward folds
and a preregistered three-value regime grid. Later phases add a selectively
invoked Kimi layer. Semantic output is evidence for a decision, never authority
to bypass risk. Exchange execution remains out of scope until shadow and demo
validation gates are met.

# Contributing

Use Python 3.12 and an isolated virtual environment. Install with
`python -m pip install -e '.[dev]'`, then run `ruff check src tests examples`,
`mypy src`, and `pytest` before opening a pull request.

Changes to strategy or feature code must include a causality test. Changes to
risk limits must include an independent veto test and demonstrate that approved
size never increases. Replay expectations are regression contracts: do not relax
one merely to make a result pass.

Keep dependencies minimal and never commit credentials, account data, external
archives, or generated position state. No live-order functionality belongs in
Phase 0/1.

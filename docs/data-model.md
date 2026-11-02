# Data model and causality

Kline JSONL records contain `{t,o,h,l,c,v}`. Desk kline timestamps are Unix
seconds at bar open. Flow records contain `{t,cls,side,price,vol}` and use Unix
milliseconds. Connectors convert both to timezone-aware UTC datetimes and reject
malformed records with file/line context. The BTC connector reads five-minute
OHLCV parquet columns without modifying the source archive.

A resample bin `[start, end)` is labeled `end` and is emitted only when all
expected source bars (one-minute or five-minute) exist. Queries at `as_of` select
labels `<= as_of`.
Consequently an indicator value at time `t` cannot observe a bar that closes
after `t`. Legacy raw flow windows are left-open/right-closed `(as_of-window,
as_of]`; P2a feature aggregation uses right-exclusive event bins `[open, close)`.
Delta/CVD, cls5/cls6 intensity, sweeps, and divergence are computed only from
current and past events. Tick aggressor is `buy` when `is_buyer_maker` is false.

The archive is opened read-only. Tests prove bin boundaries, incomplete-bin
omission, and feature invariance when future bars change.

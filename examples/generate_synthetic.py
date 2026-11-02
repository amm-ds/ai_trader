#!/usr/bin/env python3
"""Generate deterministic one-minute OHLCV JSONL for local experiments."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    """Write a synthetic fixture without third-party dependencies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=120)
    parser.add_argument("--start", type=int, default=1767225600)
    args = parser.parse_args()
    if args.bars <= 0:
        parser.error("--bars must be positive")
    price = 100.0
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.bars):
            open_ = price
            close = open_ + 0.05 + 0.03 * math.sin(index / 5)
            high = max(open_, close) + 0.04
            low = min(open_, close) - 0.04
            row = {
                "t": args.start + index * 60,
                "o": round(open_, 6),
                "h": round(high, 6),
                "l": round(low, 6),
                "c": round(close, 6),
                "v": 1000 + index,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            price = close


if __name__ == "__main__":
    main()

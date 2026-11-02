"""Resumable, daily-chunked materialization of the read-only tick lake."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ai_trader.features.micro import micro_features, normalize_ticks

_COLUMNS = ["ts_ms", "timestamp", "side", "is_buyer_maker", "price", "qty", "amount", "quote_qty", "cost"]


def read_day(path: Path) -> pd.DataFrame:
    """Read one daily parquet with schema-tolerant normalized columns."""

    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in _COLUMNS if column in available]
    return normalize_ticks(pd.read_parquet(path, columns=columns))


def materialize(symbol: str, lake_root: Path, output_root: Path) -> dict[str, float | int | str]:
    """Materialize daily chunks, resume prior progress, then atomically consolidate."""

    started = time.monotonic()
    source = lake_root / symbol
    paths = sorted(source.glob("*.parquet"))
    if not paths:
        raise ValueError(f"no tick files for {symbol}")
    chunks = output_root / "_chunks" / symbol
    progress_dir = output_root / "_progress"
    chunks.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"{symbol}.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    cvd = float(progress.get("cvd", 0.0))
    total_ticks = int(progress.get("ticks", 0))
    completed = int(progress.get("completed", 0))
    for index, path in enumerate(paths, start=1):
        day = path.stem
        chunk = chunks / f"{day}.parquet"
        if chunk.exists() and day in progress.get("days", []):
            continue
        events = read_day(path)
        features, cvd = micro_features(events, cvd_start=cvd)
        total_ticks += len(events)
        features.to_parquet(chunk, index=True)
        completed += 1
        days = progress.get("days", [])
        days.append(day)
        progress = {
            "symbol": symbol,
            "days": days,
            "completed": completed,
            "ticks": total_ticks,
            "cvd": cvd,
            "updated_at": datetime.now().isoformat(),
        }
        progress_path.write_text(json.dumps(progress, sort_keys=True) + "\n")
        if index % 25 == 0 or index == len(paths):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"{symbol} {index}/{len(paths)} chunks={completed} ticks={total_ticks} "
                f"files/s={index / elapsed:.2f} ticks/s={total_ticks / elapsed:.0f}",
                flush=True,
            )
    output = output_root / f"{symbol}_15m_features.parquet"
    dataset = ds.dataset(list(chunks.glob("*.parquet")), format="parquet")
    table = dataset.to_table()
    temporary = output.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary)
    temporary.replace(output)
    elapsed = max(time.monotonic() - started, 1e-6)
    stats = {
        "symbol": symbol,
        "files": len(paths),
        "chunks": completed,
        "ticks": total_ticks,
        "rows": table.num_rows,
        "elapsed_seconds": elapsed,
        "files_per_second": len(paths) / elapsed,
        "ticks_per_second": total_ticks / elapsed,
        "output": str(output),
    }
    (progress_dir / f"{symbol}_complete.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def tick_sanity(path: Path) -> dict[str, float | bool | str]:
    """Require exact direct-versus-pipeline delta/CVD/imbalance on one held-out day."""

    import numpy as np

    events = read_day(path)
    features, _ = micro_features(events)
    signed = events["vol"] * np.where(events["side"] == "buy", 1.0, -1.0)
    sampled = signed.resample("15min", closed="left", label="right", origin="epoch").sum()
    direct_cvd = float(signed.sum())
    direct_delta = float(sampled.iloc[-1])
    direct_imbalance = float(signed.sum() / events["vol"].sum())
    pipeline_imbalance = float(features["delta"].sum() / features["v"].sum())
    errors = {
        "cvd": abs(float(features["cvd"].iloc[-1]) - direct_cvd),
        "delta_15m": abs(float(features["delta"].iloc[-1]) - direct_delta),
        "imbalance": abs(pipeline_imbalance - direct_imbalance),
    }
    return {
        "path": str(path),
        "cvd_error": errors["cvd"],
        "delta_error": errors["delta_15m"],
        "imbalance_error": errors["imbalance"],
        "passed": all(error < 1e-8 for error in errors.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--lake-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sanity-day", type=Path)
    args = parser.parse_args()
    if args.sanity_day:
        print(json.dumps(tick_sanity(args.sanity_day), indent=2))
    else:
        print(json.dumps(materialize(args.symbol, args.lake_root, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

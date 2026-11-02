"""Read-only JSONL archive parsers with strict record validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

_KLINE_FIELDS = ("t", "o", "h", "l", "c", "v")
_FLOW_FIELDS = ("t", "cls", "side", "price", "vol")


def _records(path: Path, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record is not an object")
                row = {field: value[field] for field in fields}
                if any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for key, item in row.items()
                    if key != "side"
                ):
                    raise TypeError("numeric field is not a number")
                if not all(math.isfinite(float(item)) for key, item in row.items() if key != "side"):
                    raise ValueError("non-finite numeric field")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def load_klines(path: str | Path) -> pd.DataFrame:
    """Load `{t,o,h,l,c,v}` JSONL, indexing UTC bar-open timestamps."""

    source = Path(path)
    rows = _records(source, _KLINE_FIELDS)
    frame = pd.DataFrame(rows, columns=_KLINE_FIELDS)
    if frame.empty:
        frame.index = pd.DatetimeIndex([], tz="UTC", name="t")
        return frame.astype({field: "float64" for field in _KLINE_FIELDS[1:]})
    frame["t"] = pd.to_datetime(frame["t"].astype("int64"), unit="s", utc=True)
    frame = frame.set_index("t").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"duplicate kline timestamps in {source}")
    return frame.astype("float64")


def load_flow(path: str | Path) -> pd.DataFrame:
    """Load `{t,cls,side,price,vol}` JSONL, indexing UTC event timestamps."""

    source = Path(path)
    rows = _records(source, _FLOW_FIELDS)
    frame = pd.DataFrame(rows, columns=_FLOW_FIELDS)
    if frame.empty:
        frame.index = pd.DatetimeIndex([], tz="UTC", name="t")
        return frame
    if not frame["side"].isin(("buy", "sell")).all():
        raise ValueError(f"invalid flow side in {source}; expected 'buy' or 'sell'")
    frame["t"] = pd.to_datetime(frame["t"].astype("int64"), unit="ms", utc=True)
    frame = frame.set_index("t").sort_index()
    return frame.astype({"cls": "int64", "side": "str", "price": "float64", "vol": "float64"})

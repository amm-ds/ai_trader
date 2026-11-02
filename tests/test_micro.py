from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from ai_trader.data.ticks import materialize, tick_sanity
from ai_trader.features.micro import micro_features, normalize_ticks

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def events(count=30):
    index = pd.date_range(NOW, periods=count, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "side": np.where(np.arange(count) % 3 == 0, "sell", "buy"),
            "price": 100 + np.arange(count) * 0.1,
            "vol": np.ones(count),
            "dollars": (100 + np.arange(count) * 0.1),
        },
        index=index,
    )


def test_micro_features_are_right_exclusive_and_causal() -> None:
    first, _ = micro_features(events())
    future = events().assign(price=200, vol=100)
    future.index = future.index + timedelta(hours=1)
    second, _ = micro_features(pd.concat([events(), future]))
    assert first["delta"].iloc[0] == 5
    assert first["cvd"].iloc[-1] == 10
    assert first["ofi_15m"].iloc[0] == 1 / 3
    pdt.assert_frame_equal(first, second.iloc[:2], check_freq=False)


def test_timestamp_naming_and_missing_timestamp_handling() -> None:
    frame = pd.DataFrame(
        {"timestamp": [NOW, None], "side": ["buy", "sell"], "price": [100, 101], "amount": [2, 3], "cost": [200, 303]}
    )
    result = normalize_ticks(frame)
    assert len(result) == 1
    assert result.iloc[0]["vol"] == 2
    assert result.iloc[0]["dollars"] == 200
    seconds = pd.DataFrame(
        {
            "ts_ms": [1793608200, 1793608200000],
            "side": ["buy", "buy"],
            "price": [100, 100],
            "qty": [1, 1],
            "quote_qty": [100, 100],
        }
    )
    normalized = normalize_ticks(seconds)
    assert normalized.index[0] == datetime(2026, 11, 2, 8, 30, tzinfo=UTC)
    assert normalized.index[1] == normalized.index[0]


def test_tick_sanity_and_resumable_materialization(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    output = tmp_path / "output"
    day = lake / "BTC_USDT" / "2026-01-01.parquet"
    day.parent.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "ts_ms": [1793608200000, 1793608201000],
            "side": ["buy", "sell"],
            "price": [100.0, 100.0],
            "qty": [2.0, 1.0],
            "quote_qty": [200.0, 100.0],
            "is_buyer_maker": [False, True],
        }
    )
    raw.to_parquet(day, index=False)
    stats = materialize("BTC_USDT", lake, output)
    again = materialize("BTC_USDT", lake, output)
    assert stats["chunks"] == again["chunks"] == 1
    assert stats["rows"] == 1
    assert (output / "BTC_USDT_15m_features.parquet").exists()
    sanity = tick_sanity(day)
    assert sanity["passed"]

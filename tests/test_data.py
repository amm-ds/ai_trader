from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ai_trader.data.archive import PerSymbolView
from ai_trader.data.connectors import load_klines
from ai_trader.features.band import add_band_features
from tests.helpers import flow, klines

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_resample_is_right_exclusive_and_complete() -> None:
    view = PerSymbolView("TEST", klines(10), flow([]))
    bars = view.bars("5m")
    assert list(bars.index) == [NOW + timedelta(minutes=5), NOW + timedelta(minutes=10)]
    first = bars.iloc[0]
    assert first["o"] == 100.0 - 0.005
    assert first["c"] == 100.04
    assert first["h"] == 100.06
    assert first["v"] == 5000.0
    assert view.bars("5m", as_of=NOW + timedelta(minutes=4, seconds=59)).empty
    assert len(view.bars("5m", as_of=NOW + timedelta(minutes=5))) == 1


def test_incomplete_resample_bin_is_omitted() -> None:
    view = PerSymbolView("TEST", klines(9), flow([]))
    assert len(view.bars("5m")) == 1


def test_flow_window_excludes_left_and_includes_right() -> None:
    events = flow(
        [
            (NOW, "buy", 1.0),
            (NOW + timedelta(minutes=1), "buy", 2.0),
            (NOW + timedelta(minutes=15), "sell", 0.5),
        ]
    )
    view = PerSymbolView("TEST", klines(1), events)
    window = view.flow_window(NOW + timedelta(minutes=15), timedelta(minutes=15))
    assert window.events == 2
    assert window.buy_volume == 2.0
    assert window.sell_volume == 0.5
    assert window.delta == 1.5


def test_feature_at_t_is_invariant_to_future_changes() -> None:
    bars = klines(300)
    original = add_band_features(bars)
    changed = bars.copy()
    changed.iloc[200:, changed.columns.get_loc("c")] *= 100
    changed.iloc[200:, changed.columns.get_loc("h")] *= 100
    changed.iloc[200:, changed.columns.get_loc("l")] *= 100
    recomputed = add_band_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:200], recomputed.iloc[:200])


def test_feature_at_t_matches_recompute_using_only_history() -> None:
    bars = klines(300)
    full = add_band_features(bars)
    for cut in (59, 60, 100, 250):
        historical = add_band_features(bars.iloc[:cut])
        pd.testing.assert_series_equal(full.iloc[cut - 1], historical.iloc[-1])


def test_committed_fixture_loads_for_ci() -> None:
    fixture = Path(__file__).parents[1] / "examples" / "fixtures" / "klines.jsonl"
    frame = load_klines(fixture)
    assert len(frame) == 8
    assert str(frame.index.tz) == "UTC"
    assert frame.iloc[-1]["c"] == 100.8

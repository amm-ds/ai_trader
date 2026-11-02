from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pandas.testing as pdt

from ai_trader.features.flow_features import add_flow_features, tick_events

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def event(minute: int, side: str, price: float, vol: float, cls: int = 0):
    return {
        "t": NOW + timedelta(minutes=minute),
        "cls": cls,
        "side": side,
        "price": price,
        "vol": vol,
    }


def frame(events):
    result = pd.DataFrame(events).set_index("t")
    result.index = pd.DatetimeIndex(result.index, tz="UTC")
    return result


def bars(count=3):
    return pd.DataFrame(
        {"c": np.arange(count, dtype=float) + 100},
        index=pd.date_range(NOW + timedelta(minutes=15), periods=count, freq="15min", tz="UTC"),
    )


def test_right_exclusive_delta_cvd_and_size_weighted_log_odds() -> None:
    events = frame([
        event(1, "buy", 100, 10, 5),
        event(14, "sell", 100, 2, 6),
        event(15, "buy", 101, 5, 5),
    ])
    result = add_flow_features(events, bars())
    assert result["delta_15m"].iloc[0] == 8
    assert result["delta_15m"].iloc[1] == 5
    assert result["cvd"].iloc[1] == 13
    assert result["cls5_count_2h"].iloc[1] == 2
    assert result["cls6_count_2h"].iloc[1] == 1
    expected = np.log(15 / 2)
    assert abs(result["absorption_log_odds_2h"].iloc[1] - expected) < 1e-12


def test_future_events_do_not_change_completed_features() -> None:
    early = frame([event(1, "buy", 100, 10, 5), event(14, "sell", 100, 2, 6)])
    extended = frame([
        event(1, "buy", 100, 10, 5),
        event(14, "sell", 100, 2, 6),
        event(30, "buy", 105, 50, 5),
    ])
    first = add_flow_features(early, bars(2))
    second = add_flow_features(extended, bars(3)).iloc[:2]
    pdt.assert_frame_equal(first, second)


def test_sweep_uses_large_print_crossing_prior_twenty_print_extreme() -> None:
    events = [event(index, "buy", 100, 1) for index in range(20)]
    events.append(event(20, "buy", 101, 5))
    result = add_flow_features(frame(events), bars(2))
    assert result["sweep_buy_1h"].iloc[1] == 1


def test_tick_aggressor_mapping_uses_buyer_maker() -> None:
    ticks = pd.DataFrame(
        {"ts_ms": [1, 2], "price": [100, 101], "qty": [2, 3], "is_buyer_maker": [False, True]}
    )
    result = tick_events(ticks)
    assert result["side"].tolist() == ["buy", "sell"]
    assert result["vol"].sum() == 5

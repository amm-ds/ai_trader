"""Causal, bar-close-aligned order-flow features from desk or tick events."""

from __future__ import annotations

import numpy as np
import pandas as pd

SWEEP_VOLUME_MULTIPLE = 3.0
SWEEP_PRINTS = 20
_PERCENTILE_WINDOW = 1920
_EPSILON = 1e-12


def add_flow_features(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate right-exclusive `[open, close)` events onto completed bar closes.

    Required event columns are `side`, `price`, and `vol`; `cls` is optional for
    tick-derived frames. All rolling windows use only the current and past bars.
    """

    if bars.empty:
        return pd.DataFrame(index=bars.index)
    close = bars["c"].astype("float64")
    if events.empty:
        sampled = pd.DataFrame(index=close.index)
        for column in ("signed", "cls5_count", "cls6_count", "cls5_vol", "cls6_vol", "sweep_buy", "sweep_sell"):
            sampled[column] = 0.0
        cvd = pd.Series(0.0, index=close.index)
    else:
        frame = events.copy().sort_index()
        frame["signed"] = np.where(frame["side"] == "buy", frame["vol"], -frame["vol"])
        frame["cls5_count"] = (frame["cls"] == 5).astype("float64") if "cls" in frame else 0.0
        frame["cls6_count"] = (frame["cls"] == 6).astype("float64") if "cls" in frame else 0.0
        frame["cls5_vol"] = frame["vol"] * (frame["cls"] == 5) if "cls" in frame else 0.0
        frame["cls6_vol"] = frame["vol"] * (frame["cls"] == 6) if "cls" in frame else 0.0
        prior_median = frame["vol"].rolling(100, min_periods=20).median().shift(1)
        prior_high = frame["price"].rolling(SWEEP_PRINTS, min_periods=SWEEP_PRINTS).max().shift(1)
        prior_low = frame["price"].rolling(SWEEP_PRINTS, min_periods=SWEEP_PRINTS).min().shift(1)
        large = frame["vol"] > SWEEP_VOLUME_MULTIPLE * prior_median
        frame["sweep_buy"] = ((frame["side"] == "buy") & large & (frame["price"] > prior_high)).astype("float64")
        frame["sweep_sell"] = ((frame["side"] == "sell") & large & (frame["price"] < prior_low)).astype("float64")
        columns = ("signed", "cls5_count", "cls6_count", "cls5_vol", "cls6_vol", "sweep_buy", "sweep_sell")
        sampled = frame[list(columns)].resample("15min", closed="left", label="right", origin="epoch").sum()
        sampled = sampled.reindex(close.index, fill_value=0.0)
        cvd = frame["signed"].cumsum().resample("15min", closed="left", label="right", origin="epoch").last()
        cvd = cvd.reindex(close.index).ffill().fillna(0.0)

    result = pd.DataFrame(index=close.index)
    result["delta_15m"] = sampled["signed"]
    result["cvd"] = cvd
    result["cvd_change_15m"] = cvd.diff().fillna(cvd)
    for hours, window in ((1, 4), (6, 24)):
        result[f"delta_{hours}h"] = sampled["signed"].rolling(window, min_periods=1).sum()
        result[f"cvd_change_{hours}h"] = cvd.diff(window).fillna(0.0)
        result[f"volume_{hours}h"] = sampled["signed"].abs().rolling(window, min_periods=1).sum()
        result[f"imbalance_{hours}h"] = result[f"delta_{hours}h"] / result[f"volume_{hours}h"].replace(0, np.nan)
    result["volume_15m"] = sampled["signed"].abs()
    result["imbalance_15m"] = result["delta_15m"] / result["volume_15m"].replace(0, np.nan)
    for hours, window in ((2, 8), (6, 24)):
        for class_id in (5, 6):
            result[f"cls{class_id}_count_{hours}h"] = sampled[f"cls{class_id}_count"].rolling(window, min_periods=1).sum()
            result[f"cls{class_id}_vol_{hours}h"] = sampled[f"cls{class_id}_vol"].rolling(window, min_periods=1).sum()
        count_ratio = (result[f"cls5_count_{hours}h"] + _EPSILON) / (result[f"cls6_count_{hours}h"] + _EPSILON)
        intensity_ratio = (result[f"cls5_vol_{hours}h"] + _EPSILON) / (result[f"cls6_vol_{hours}h"] + _EPSILON)
        result[f"cls5_cls6_count_ratio_{hours}h"] = count_ratio
        result[f"cls5_cls6_intensity_ratio_{hours}h"] = intensity_ratio
        result[f"count_log_odds_{hours}h"] = np.log(count_ratio)
        result[f"absorption_log_odds_{hours}h"] = np.log(intensity_ratio)
    result["sweep_buy_1h"] = sampled["sweep_buy"].rolling(4, min_periods=1).sum()
    result["sweep_sell_1h"] = sampled["sweep_sell"].rolling(4, min_periods=1).sum()
    result["cvd_slope_pct"] = result["cvd_change_15m"].abs().rolling(
        _PERCENTILE_WINDOW, min_periods=100
    ).rank(pct=True)
    price_change = close.diff(3)
    cvd_change = cvd.diff(3)
    result["bullish_delta_price_divergence"] = (price_change < 0) & (cvd_change > 0)
    result["bearish_delta_price_divergence"] = (price_change > 0) & (cvd_change < 0)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def tick_events(ticks: pd.DataFrame) -> pd.DataFrame:
    """Normalize aggressor-flagged ticks; buyer-maker means sell-initiated."""

    required = {"ts_ms", "price", "is_buyer_maker"}
    if not required.issubset(ticks.columns):
        raise ValueError(f"tick data missing {required - set(ticks.columns)}")
    volume_column = "qty" if "qty" in ticks.columns else "amount" if "amount" in ticks.columns else None
    if volume_column is None:
        raise ValueError("tick data requires qty or amount")
    frame = pd.DataFrame(
        {
            "cls": 0,
            "side": np.where(ticks["is_buyer_maker"], "sell", "buy"),
            "price": ticks["price"].to_numpy(dtype="float64"),
            "vol": ticks[volume_column].to_numpy(dtype="float64"),
        },
        index=pd.to_datetime(ticks["ts_ms"].astype("int64"), unit="ms", utc=True),
    )
    frame.index.name = "t"
    return frame.sort_index()

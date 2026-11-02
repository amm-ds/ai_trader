"""Causal 15m microstructure features from aggressor-flagged tick events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trader.features.indicators import add_strategy_features

SWEEP_VOLUME_MULTIPLE = 3.0
SWEEP_PRINTS = 20


def normalize_ticks(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp, aggressor, price, base quantity, and dollar columns."""

    if "ts_ms" in frame.columns:
        milliseconds = pd.to_numeric(frame["ts_ms"], errors="coerce")
        valid = milliseconds.notna()
        frame = frame.loc[valid]
        values = milliseconds.loc[valid].astype("int64")
        normalized_ms = values.where(values >= 100_000_000_000, values * 1000)
        index = pd.to_datetime(normalized_ms, unit="ms", utc=True)
    elif "timestamp" in frame.columns:
        valid = frame["timestamp"].notna()
        frame = frame.loc[valid]
        index = pd.to_datetime(frame["timestamp"], utc=True)
    else:
        raise ValueError("tick data requires ts_ms or timestamp")
    quantity = "qty" if "qty" in frame.columns else "amount" if "amount" in frame.columns else None
    if quantity is None:
        raise ValueError("tick data requires qty or amount")
    dollars = "quote_qty" if "quote_qty" in frame.columns else "cost" if "cost" in frame.columns else None
    if "is_buyer_maker" in frame.columns:
        side = np.where(frame["is_buyer_maker"], "sell", "buy")
    elif "side" in frame.columns:
        side = frame["side"].astype("str").str.lower().to_numpy()
    else:
        raise ValueError("tick data requires side or is_buyer_maker")
    result = pd.DataFrame(
        {
            "side": side,
            "price": frame["price"].to_numpy(dtype="float64"),
            "vol": frame[quantity].to_numpy(dtype="float64"),
            "dollars": frame[dollars].to_numpy(dtype="float64") if dollars else frame["price"].to_numpy(dtype="float64") * frame[quantity].to_numpy(dtype="float64"),
        },
        index=index,
    )
    result.index.name = "t"
    return result.sort_index()


def micro_features(events: pd.DataFrame, cvd_start: float = 0.0) -> tuple[pd.DataFrame, float]:
    """Aggregate right-exclusive 15m bars and causal rolling microstructure."""

    if events.empty:
        raise ValueError("cannot compute micro features from an empty day")
    frame = events.copy()
    frame["signed"] = np.where(frame["side"] == "buy", frame["vol"], -frame["vol"])
    frame["buy_vol"] = frame["vol"] * (frame["side"] == "buy")
    frame["sell_vol"] = frame["vol"] * (frame["side"] == "sell")
    frame["count"] = 1.0
    prior_median = frame["vol"].rolling(100, min_periods=20).median().shift(1)
    prior_high = frame["price"].rolling(SWEEP_PRINTS, min_periods=SWEEP_PRINTS).max().shift(1)
    prior_low = frame["price"].rolling(SWEEP_PRINTS, min_periods=SWEEP_PRINTS).min().shift(1)
    large = frame["vol"] > SWEEP_VOLUME_MULTIPLE * prior_median
    frame["sweep_buy"] = ((frame["side"] == "buy") & large & (frame["price"] > prior_high)).astype("float64")
    frame["sweep_sell"] = ((frame["side"] == "sell") & large & (frame["price"] < prior_low)).astype("float64")
    grouped = frame.resample("15min", closed="left", label="right", origin="epoch")
    bars = grouped.agg(
        o=("price", "first"), h=("price", "max"), l=("price", "min"), c=("price", "last"),
        v=("vol", "sum"), dollar_volume=("dollars", "sum"), trade_count=("count", "sum"),
        buy_volume=("buy_vol", "sum"), sell_volume=("sell_vol", "sum"), delta=("signed", "sum"),
        sweep_buy=("sweep_buy", "sum"), sweep_sell=("sweep_sell", "sum"),
    ).dropna(subset=["c"])
    total = bars["buy_volume"] + bars["sell_volume"]
    bars["ofi_15m"] = bars["delta"] / total.replace(0, np.nan)
    bars["buy_pressure"] = bars["buy_volume"] / total.replace(0, np.nan)
    bars["sell_pressure"] = bars["sell_volume"] / total.replace(0, np.nan)
    bars["cvd"] = cvd_start + frame["signed"].cumsum().resample(
        "15min", closed="left", label="right", origin="epoch"
    ).last().reindex(bars.index).ffill().fillna(0.0)
    bars["cvd_slope_1h"] = bars["cvd"].diff(4) / 4
    bars["microprice"] = bars["dollar_volume"] / bars["v"].replace(0, np.nan)
    deviation = (bars["microprice"] - bars["c"]) / bars["c"]
    bars["microprice_deviation_ewma"] = deviation.ewm(span=20, adjust=False, min_periods=5).mean()
    for column in ("delta", "dollar_volume", "trade_count"):
        mean = bars[column].rolling(96, min_periods=20).mean()
        std = bars[column].rolling(96, min_periods=20).std(ddof=0).replace(0, np.nan)
        name = "signed_volume_z" if column == "delta" else f"{column}_z"
        bars[name] = (bars[column] - mean) / std
    enriched = add_strategy_features(bars[["o", "h", "l", "c", "v"]])
    for column in ("ema9", "bb_mid", "ma60", "band_score", "trend", "atr14", "rsi3", "roc_1", "bb_width_pct", "realized_vol_pct"):
        bars[column] = enriched[column]
    for timeframe, rule, expected in (("1h", "1h", 4), ("4h", "4h", 16)):
        higher = bars[["o", "h", "l", "c", "v"]].resample(rule, closed="left", label="right", origin="epoch").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
        )
        count = bars["c"].resample(rule, closed="left", label="right", origin="epoch").count()
        higher = higher.loc[count == expected].dropna()
        features = add_strategy_features(higher)
        bars[f"trend_{timeframe}"] = features["trend"].reindex(bars.index).ffill().fillna("neutral")
        bars[f"return_{timeframe}"] = features["roc_1"].reindex(bars.index).ffill().fillna(0.0)
        bars[f"bb_width_pct_{timeframe}"] = features["bb_width_pct"].reindex(bars.index).ffill().fillna(0.0)
        bars[f"realized_vol_pct_{timeframe}"] = features["realized_vol_pct"].reindex(bars.index).ffill().fillna(0.0)
    bars["trend_5m"] = bars["trend"]
    bars["trend_15m"] = bars["trend"]
    bars["return_15m"] = bars["roc_1"]
    return bars.replace([np.inf, -np.inf], np.nan), float(bars["cvd"].iloc[-1])

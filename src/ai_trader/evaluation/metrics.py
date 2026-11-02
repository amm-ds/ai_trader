"""Uncertainty-aware performance statistics for daily marked backtest results."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from statistics import NormalDist
from typing import Any

import numpy as np

from ai_trader.evaluation.backtester import EntryAudit
from ai_trader.execution.paper import ClosedTrade


def daily_returns(
    trades: Sequence[ClosedTrade], start: datetime, end: datetime, initial_equity: float
) -> np.ndarray:
    """Return simple daily PnL/equity returns, including zero-trade calendar days."""

    days = (end.date() - start.date()).days
    if days <= 0 or initial_equity <= 0:
        raise ValueError("invalid metric range or equity")
    pnl: dict[date, float] = defaultdict(float)
    for trade in trades:
        pnl[trade.closed_at.date()] += trade.net_pnl
    return np.array([pnl[start.date() + timedelta(days=offset)] / initial_equity for offset in range(days)])


def sharpe_ratio(returns: np.ndarray) -> float:
    """Annualized daily Sharpe; a zero-variance series has no measured edge."""

    if len(returns) < 2:
        return 0.0
    volatility = float(np.std(returns, ddof=1))
    if not math.isfinite(volatility) or volatility <= 0:
        return 0.0
    return float(np.mean(returns) / volatility * math.sqrt(365))


def sortino_ratio(returns: np.ndarray) -> float:
    """Annualized daily Sortino using zero as the minimum acceptable return."""

    if len(returns) < 2:
        return 0.0
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    if downside <= 0:
        return 0.0
    return float(np.mean(returns) / downside * math.sqrt(365))


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    seed: int = 20261031,
    samples: int = 2000,
    block_length: int = 7,
) -> tuple[float, float]:
    """Circularly sample seven-day blocks and return a deterministic 95% CI."""

    if len(returns) < 2 or float(np.std(returns, ddof=1)) <= 0 or samples <= 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    length = min(block_length, len(returns))
    blocks_needed = math.ceil(len(returns) / length)
    starts = rng.integers(0, len(returns), size=(samples, blocks_needed))
    offsets = np.arange(length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(returns)
    bootstrapped = returns[indices].reshape(samples, blocks_needed * length)[:, : len(returns)]
    volatility = np.std(bootstrapped, axis=1, ddof=1)
    values = np.divide(
        np.mean(bootstrapped, axis=1),
        volatility,
        out=np.zeros(samples),
        where=volatility > 0,
    ) * math.sqrt(365)
    low, high = np.quantile(values, [0.025, 0.975])
    return (float(low), float(high))


def permutation_pbo(returns: np.ndarray, seed: int = 20261031, samples: int = 2000) -> float:
    """Estimate failure by random sign-permutation exceedance of observed Sharpe.

    Reordering returns alone cannot change their mean or standard deviation, so
    the null permutes each day's sign rather than pretending order affects Sharpe.
    """

    observed = sharpe_ratio(returns)
    if len(returns) < 2 or float(np.std(returns, ddof=1)) <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(samples):
        permuted = returns * rng.choice(np.array([-1.0, 1.0]), size=len(returns))
        if sharpe_ratio(permuted) >= observed:
            exceedances += 1
    return exceedances / samples


def deflated_sharpe_ratio(returns: np.ndarray, trials: int) -> float:
    """Return the probability that observed daily Sharpe exceeds expected selection noise."""

    count = len(returns)
    if count < 3 or trials < 1:
        return 0.0
    volatility = float(np.std(returns, ddof=1))
    if volatility <= 0:
        return 0.0
    sharpe = float(np.mean(returns) / volatility)
    centered = returns - np.mean(returns)
    skew = float(np.mean(centered**3) / volatility**3)
    kurtosis = float(np.mean(centered**4) / volatility**4)
    normal = NormalDist()
    euler = 0.5772156649015329
    if trials == 1:
        expected_max = 0.0
    else:
        expected_max = volatility * (
            (1 - euler) * normal.inv_cdf(1 - 1 / trials)
            + euler * normal.inv_cdf(1 - 1 / (trials * math.e))
        )
    denominator = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if denominator <= 0:
        return 0.0
    z_score = (sharpe - expected_max) * math.sqrt(count - 1) / math.sqrt(denominator)
    return normal.cdf(z_score)


def performance_metrics(
    trades: Sequence[ClosedTrade],
    entries: Sequence[EntryAudit],
    start: datetime,
    end: datetime,
    initial_equity: float = 100.0,
    symbol_count: int = 1,
    trials: int = 9,
) -> dict[str, Any]:
    """Compute the preregistered headline table and per-regime attribution."""

    returns = daily_returns(trades, start, end, initial_equity)
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = -sum(trade.net_pnl for trade in losses)
    pnl_by_day: dict[date, float] = defaultdict(float)
    for trade in trades:
        pnl_by_day[trade.closed_at.date()] += trade.net_pnl
    equity = initial_equity
    peak = initial_equity
    max_drawdown = 0.0
    for offset in range(len(returns)):
        equity += pnl_by_day[start.date() + timedelta(days=offset)]
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    duration = (end - start).total_seconds()
    exposure = (
        sum((trade.closed_at - trade.opened_at).total_seconds() for trade in trades)
        / (duration * symbol_count)
        if duration > 0 and symbol_count > 0
        else 0.0
    )
    turnover = sum(
        trade.quantity * (trade.entry_price + trade.exit_price) for trade in trades
    ) / initial_equity
    ci_low, ci_high = block_bootstrap_sharpe_ci(returns)
    attribution: dict[str, dict[str, float | int]] = {}
    strategy_attribution: dict[str, dict[str, dict[str, float | int]]] = {}
    entries_by_id = {entry.position_id: entry for entry in entries}
    for trade in trades:
        entry = entries_by_id.get(trade.position_id)
        regime = entry.regime if entry is not None and entry.regime else "UNKNOWN"
        strategy = entry.strategy if entry is not None and entry.strategy else "unknown"
        row = attribution.setdefault(regime, {"trades": 0, "net_pnl": 0.0, "sum_r": 0.0})
        row["trades"] = int(row["trades"]) + 1
        row["net_pnl"] = float(row["net_pnl"]) + trade.net_pnl
        row["sum_r"] = float(row["sum_r"]) + trade.r_multiple
        strategy_row = strategy_attribution.setdefault(strategy, {}).setdefault(
            regime,
            {
                "trades": 0,
                "wins": 0,
                "net_pnl": 0.0,
                "sum_r": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            },
        )
        strategy_row["trades"] = int(strategy_row["trades"]) + 1
        strategy_row["wins"] = int(strategy_row["wins"]) + int(trade.net_pnl > 0)
        strategy_row["net_pnl"] = float(strategy_row["net_pnl"]) + trade.net_pnl
        strategy_row["sum_r"] = float(strategy_row["sum_r"]) + trade.r_multiple
        if trade.net_pnl > 0:
            strategy_row["gross_profit"] = float(strategy_row["gross_profit"]) + trade.net_pnl
        elif trade.net_pnl < 0:
            strategy_row["gross_loss"] = float(strategy_row["gross_loss"]) - trade.net_pnl
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "expectancy_r": sum(trade.r_multiple for trade in trades) / len(trades) if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        "sharpe_daily": sharpe_ratio(returns),
        "sortino_daily": sortino_ratio(returns),
        "max_drawdown": max_drawdown,
        "exposure": exposure,
        "turnover": turnover,
        "net_pnl": sum(trade.net_pnl for trade in trades),
        "sharpe_ci_95": [ci_low, ci_high],
        "pbo_permutation": permutation_pbo(returns),
        "deflated_sharpe": deflated_sharpe_ratio(returns, trials=trials),
        "regime_attribution": attribution,
        "strategy_attribution": strategy_attribution,
    }

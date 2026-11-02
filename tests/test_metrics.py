from datetime import UTC, datetime

import numpy as np
import pytest

from ai_trader.evaluation.metrics import (
    block_bootstrap_sharpe_ci,
    daily_returns,
    deflated_sharpe_ratio,
    permutation_pbo,
    sharpe_ratio,
    sortino_ratio,
)
from ai_trader.evaluation.walk_forward import walk_forward_folds

START = datetime(2026, 1, 1, tzinfo=UTC)


def test_daily_sharpe_and_sortino_match_direct_math() -> None:
    returns = np.array([0.01, -0.005, 0.02, 0.0])
    expected = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365)
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    assert sharpe_ratio(returns) == pytest.approx(expected)
    assert sortino_ratio(returns) == pytest.approx(np.mean(returns) / downside * np.sqrt(365))


def test_zero_return_days_are_included() -> None:
    returns = daily_returns([], START, START.replace(day=4), 100)
    np.testing.assert_array_equal(returns, np.zeros(3))


def test_block_bootstrap_ci_is_deterministic_with_fixed_seed() -> None:
    returns = np.array([0.01, -0.004, 0.006, 0.0, -0.002, 0.008] * 4)
    first = block_bootstrap_sharpe_ci(returns, seed=42, samples=500)
    second = block_bootstrap_sharpe_ci(returns, seed=42, samples=500)
    assert first == second
    assert first[0] <= sharpe_ratio(returns) <= first[1]
    assert block_bootstrap_sharpe_ci(np.zeros(10)) == (0.0, 0.0)


def test_permutation_pbo_and_deflated_sharpe_penalize_noise() -> None:
    positive = np.array([0.01, -0.002, 0.008, 0.004, -0.001] * 10)
    negative = -positive
    assert permutation_pbo(positive, samples=500) < 0.1
    assert permutation_pbo(negative, samples=500) > 0.9
    assert deflated_sharpe_ratio(positive, trials=3) > deflated_sharpe_ratio(negative, trials=3)
    assert deflated_sharpe_ratio(positive, trials=9) < deflated_sharpe_ratio(positive, trials=1)


def test_walk_forward_tests_are_later_and_contiguous() -> None:
    folds = walk_forward_folds(START, START.replace(month=2))
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_end == fold.test_start
        assert fold.train_start == START
        assert fold.test_start < fold.test_end
    assert folds[0].test_end == folds[1].test_start
    assert folds[1].test_end == folds[2].test_start

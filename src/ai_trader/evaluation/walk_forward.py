"""Three expanding-window folds with six contiguous, non-overlapping time segments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Fold:
    """One expanding train interval and its strictly later test interval."""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def walk_forward_folds(start: datetime, end: datetime) -> tuple[Fold, Fold, Fold]:
    """Use 3/6, 4/6, and 5/6 as train cutoffs; test the following sixth."""

    if start >= end:
        raise ValueError("invalid walk-forward range")
    segment = (end - start) / 6
    folds = []
    for index in range(3):
        cutoff = start + segment * (index + 3)
        folds.append(
            Fold(index + 1, start, cutoff, cutoff, cutoff + segment)
        )
    return tuple(folds)  # type: ignore[return-value]

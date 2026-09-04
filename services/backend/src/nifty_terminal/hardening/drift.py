"""Small deterministic drift metrics for explicit reference distributions."""

from __future__ import annotations

from math import log
from typing import Sequence


def population_stability_index(reference: Sequence[float], current: Sequence[float]) -> float:
    left, right = _normalized_pair(reference, current)
    return sum((observed - expected) * log(observed / expected) for expected, observed in zip(left, right))


def jensen_shannon_divergence(reference: Sequence[float], current: Sequence[float]) -> float:
    left, right = _normalized_pair(reference, current)
    midpoint = tuple((a + b) / 2 for a, b in zip(left, right))
    return (_kl(left, midpoint) + _kl(right, midpoint)) / 2


def _normalized_pair(reference: Sequence[float], current: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(reference) != len(current) or len(reference) < 2:
        raise ValueError("Drift distributions must have the same number of at least two bins")
    if any(value < 0 for value in (*reference, *current)):
        raise ValueError("Drift distributions cannot contain negative values")
    return _normalize(reference), _normalize(current)


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    epsilon = 1e-12
    total = sum(values)
    if total <= 0:
        raise ValueError("Drift distribution must contain positive mass")
    smoothed = tuple((value / total) + epsilon for value in values)
    denominator = sum(smoothed)
    return tuple(value / denominator for value in smoothed)


def _kl(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * log(a / b) for a, b in zip(left, right))

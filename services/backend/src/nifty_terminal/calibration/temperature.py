"""Numerically small, deterministic multiclass temperature scaling."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-12


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive")
    if probabilities.ndim != 2:
        raise ValueError("Probability input must be a matrix")
    clipped = np.clip(probabilities, EPSILON, 1.0)
    logits = np.log(clipped) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, actual_indices: np.ndarray) -> float:
    if len(probabilities) != len(actual_indices) or not len(actual_indices):
        raise ValueError("Temperature fitting requires aligned observations")

    def objective(log_temperature: float) -> float:
        calibrated = apply_temperature(probabilities, float(np.exp(log_temperature)))
        selected = calibrated[np.arange(len(actual_indices)), actual_indices]
        return float(-np.mean(np.log(np.clip(selected, EPSILON, 1.0))))

    # Golden-section search in log-temperature space avoids a SciPy dependency.
    lower, upper = float(np.log(0.05)), float(np.log(20.0))
    ratio = (5**0.5 - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_score, right_score = objective(left), objective(right)
    for _ in range(96):
        if left_score <= right_score:
            upper, right, right_score = right, left, left_score
            left = upper - ratio * (upper - lower)
            left_score = objective(left)
        else:
            lower, left, left_score = left, right, right_score
            right = lower + ratio * (upper - lower)
            right_score = objective(right)
    return float(np.exp((lower + upper) / 2))

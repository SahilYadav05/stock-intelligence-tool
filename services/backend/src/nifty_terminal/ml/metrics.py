"""Transparent multiclass metrics for uncalibrated research predictions."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import log_loss

from nifty_terminal.ml.definitions import CLASS_ORDER
from nifty_terminal.ml.models import MetricSummary


def calculate_metrics(
    actual: tuple[str, ...],
    probabilities: np.ndarray,
) -> MetricSummary:
    if not actual:
        raise ValueError("Metrics require at least one prediction")
    if probabilities.shape != (len(actual), len(CLASS_ORDER)):
        raise ValueError("Probability matrix does not match samples and class order")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Every probability row must sum to one")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must remain inside [0, 1]")

    actual_array = np.asarray(actual)
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.asarray([CLASS_ORDER[index] for index in predicted_indices])
    support = Counter(actual)
    recalls: list[tuple[str, float]] = []
    supported_recalls: list[float] = []
    for class_name in CLASS_ORDER:
        mask = actual_array == class_name
        recall = float(np.mean(predicted[mask] == class_name)) if np.any(mask) else 0.0
        recalls.append((class_name, recall))
        if np.any(mask):
            supported_recalls.append(recall)

    one_hot = np.zeros_like(probabilities)
    class_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    for row_index, class_name in enumerate(actual):
        one_hot[row_index, class_index[class_name]] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(predicted == actual_array))
    return MetricSummary(
        sample_count=len(actual),
        accuracy=accuracy,
        balanced_accuracy=float(np.mean(supported_recalls)),
        multiclass_brier=brier,
        log_loss=float(log_loss(actual, probabilities, labels=list(CLASS_ORDER))),
        raw_ece_10_bin=_expected_calibration_error(actual_array, predicted, probabilities),
        class_support=tuple((name, support[name]) for name in CLASS_ORDER),
        class_recall=tuple(recalls),
    )


def prior_probabilities(
    training_actual: tuple[str, ...],
    prediction_count: int,
) -> np.ndarray:
    if not training_actual:
        raise ValueError("A prior baseline requires training labels")
    support = Counter(training_actual)
    row = np.asarray([support[name] / len(training_actual) for name in CLASS_ORDER])
    return np.tile(row, (prediction_count, 1))


def _expected_calibration_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    total = len(actual)
    error = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        mask = (confidence >= lower) & (
            confidence <= upper if bin_index == 9 else confidence < upper
        )
        if not np.any(mask):
            continue
        error += float(np.sum(mask) / total) * abs(
            float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
        )
    return error

"""Fit on earlier OOS predictions and evaluate on an untouched later block."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from nifty_terminal.calibration.definitions import (
    CALIBRATION_IDENTITY,
    CALIBRATION_VERSION,
)
from nifty_terminal.calibration.models import (
    CalibratedPrediction,
    CalibrationConfig,
    CalibrationObservation,
    CalibrationReport,
    SliceGate,
    TemperatureArtifact,
)
from nifty_terminal.calibration.temperature import apply_temperature, fit_temperature
from nifty_terminal.ml.definitions import CLASS_ORDER
from nifty_terminal.ml.metrics import calculate_metrics, prior_probabilities


class CalibrationPipeline:
    def run(
        self,
        *,
        observations: tuple[CalibrationObservation, ...],
        config: CalibrationConfig | None = None,
        created_at: datetime | None = None,
    ) -> CalibrationReport:
        resolved = config or CalibrationConfig()
        ordered = tuple(sorted(observations, key=lambda item: (item.decision_time, item.prediction_id)))
        if not ordered:
            raise ValueError("Calibration requires chronological OOS predictions")
        if len({item.run_id for item in ordered}) != 1:
            raise ValueError("Calibration observations must belong to one research run")
        if len({item.candidate_name for item in ordered}) != 1:
            raise ValueError("Calibration observations must belong to one candidate")
        if len({item.prediction_id for item in ordered}) != len(ordered):
            raise ValueError("Calibration observations contain duplicate prediction IDs")

        split_index = int(len(ordered) * resolved.fit_fraction)
        if split_index < 1 or split_index >= len(ordered):
            raise ValueError("Calibration split produced an empty partition")
        fit_rows, evaluation_rows = ordered[:split_index], ordered[split_index:]
        if fit_rows[-1].decision_time >= evaluation_rows[0].decision_time:
            raise ValueError("Calibration fit and evaluation partitions must be chronological")

        fit_probabilities = _matrix(fit_rows)
        evaluation_raw = _matrix(evaluation_rows)
        fit_actual = tuple(item.actual_outcome.value for item in fit_rows)
        evaluation_actual = tuple(item.actual_outcome.value for item in evaluation_rows)
        class_index = {name: index for index, name in enumerate(CLASS_ORDER)}
        actual_indices = np.asarray([class_index[name] for name in fit_actual], dtype=int)
        temperature = fit_temperature(fit_probabilities, actual_indices)
        evaluation_calibrated = apply_temperature(evaluation_raw, temperature)
        fit_prior = prior_probabilities(fit_actual, len(evaluation_rows))

        raw_metrics = calculate_metrics(evaluation_actual, evaluation_raw)
        calibrated_metrics = calculate_metrics(evaluation_actual, evaluation_calibrated)
        prior_metrics = calculate_metrics(evaluation_actual, fit_prior)
        brier_skill = 1.0 - (
            calibrated_metrics.multiclass_brier / prior_metrics.multiclass_brier
        )
        supported_bins = _supported_bins(
            evaluation_calibrated,
            resolved.minimum_supported_probability_bin,
        )
        slices = _slice_gates(
            rows=evaluation_rows,
            probabilities=evaluation_calibrated,
            prior=fit_prior,
            config=resolved,
        )
        blockers = _release_blockers(
            config=resolved,
            all_rows=ordered,
            fit_rows=fit_rows,
            evaluation_rows=evaluation_rows,
            metrics=calibrated_metrics,
            raw_metrics=raw_metrics,
            brier_skill=brier_skill,
            supported_bins=supported_bins,
            slice_gates=slices,
        )
        timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        calibration_id = _calibration_id(ordered, resolved, timestamp)
        artifact = TemperatureArtifact(
            schema_version=1,
            calibration_id=calibration_id,
            source_run_id=ordered[0].run_id,
            candidate_name=ordered[0].candidate_name,
            calibration_version=CALIBRATION_VERSION,
            calibration_identity=CALIBRATION_IDENTITY,
            fitted_at=timestamp,
            fit_ends_at=fit_rows[-1].decision_time,
            evaluation_starts_at=evaluation_rows[0].decision_time,
            temperature=temperature,
        )
        predictions = tuple(
            CalibratedPrediction(
                calibrated_prediction_id=str(
                    uuid5(NAMESPACE_URL, f"calibrated:{calibration_id}:{row.prediction_id}")
                ),
                calibration_id=calibration_id,
                source_prediction_id=row.prediction_id,
                decision_time=row.decision_time,
                calibrated_probabilities=tuple(
                    (name, float(evaluation_calibrated[index, class_index[name]]))
                    for name in CLASS_ORDER
                ),
                actual_outcome=row.actual_outcome,
                evaluation_partition=True,
            )
            for index, row in enumerate(evaluation_rows)
        )
        return CalibrationReport(
            schema_version=1,
            calibration_id=calibration_id,
            created_at=timestamp,
            artifact=artifact,
            config=resolved,
            total_observations=len(ordered),
            fit_observations=len(fit_rows),
            evaluation_observations=len(evaluation_rows),
            raw_evaluation_metrics=raw_metrics,
            calibrated_evaluation_metrics=calibrated_metrics,
            prior_evaluation_metrics=prior_metrics,
            brier_skill=brier_skill,
            supported_confidence_bins=supported_bins,
            slice_gates=slices,
            release_gate_passed=not blockers,
            blockers=blockers,
            predictions=predictions,
        )


def _matrix(rows: tuple[CalibrationObservation, ...]) -> np.ndarray:
    matrix = np.asarray(
        [[dict(item.raw_probabilities)[name] for name in CLASS_ORDER] for item in rows],
        dtype=float,
    )
    if matrix.shape != (len(rows), len(CLASS_ORDER)):
        raise ValueError("Every calibration row must contain the complete class order")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Calibration probability rows must sum to one")
    return matrix


def _supported_bins(probabilities: np.ndarray, minimum_support: int) -> tuple[str, ...]:
    confidence = probabilities.max(axis=1)
    supported = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        upper_mask = confidence <= upper if index == 9 else confidence < upper
        count = int(np.sum((confidence >= lower) & upper_mask))
        if count >= minimum_support:
            supported.append(f"{lower:.1f}-{upper:.1f}")
    return tuple(supported)


def _slice_gates(
    *,
    rows: tuple[CalibrationObservation, ...],
    probabilities: np.ndarray,
    prior: np.ndarray,
    config: CalibrationConfig,
) -> tuple[SliceGate, ...]:
    fold_ids = sorted({item.fold_index for item in rows})
    gates = []
    for fold_id in fold_ids:
        indices = [index for index, item in enumerate(rows) if item.fold_index == fold_id]
        if len(indices) < config.minimum_slice_samples:
            gates.append(SliceGate(f"fold-{fold_id}", len(indices), None, None, False))
            continue
        actual = tuple(rows[index].actual_outcome.value for index in indices)
        metrics = calculate_metrics(actual, probabilities[indices])
        prior_metrics = calculate_metrics(actual, prior[indices])
        skill = 1.0 - metrics.multiclass_brier / prior_metrics.multiclass_brier
        passed = (
            metrics.raw_ece_10_bin <= config.maximum_slice_ece
            and skill >= config.minimum_slice_brier_skill
        )
        gates.append(SliceGate(f"fold-{fold_id}", len(indices), metrics.raw_ece_10_bin, skill, passed))
    return tuple(gates)


def _release_blockers(
    *,
    config: CalibrationConfig,
    all_rows: tuple[CalibrationObservation, ...],
    fit_rows: tuple[CalibrationObservation, ...],
    evaluation_rows: tuple[CalibrationObservation, ...],
    metrics,
    raw_metrics,
    brier_skill: float,
    supported_bins: tuple[str, ...],
    slice_gates: tuple[SliceGate, ...],
) -> tuple[str, ...]:
    blockers = []
    if len(all_rows) < config.minimum_total_predictions:
        blockers.append("INSUFFICIENT_TOTAL_OOS_PREDICTIONS")
    fit_support = Counter(item.actual_outcome.value for item in fit_rows)
    evaluation_support = Counter(item.actual_outcome.value for item in evaluation_rows)
    if any(fit_support[name] < config.minimum_fit_class_support for name in CLASS_ORDER):
        blockers.append("INSUFFICIENT_CALIBRATION_FIT_CLASS_SUPPORT")
    if any(evaluation_support[name] < config.minimum_evaluation_class_support for name in CLASS_ORDER):
        blockers.append("INSUFFICIENT_UNTOUCHED_EVALUATION_CLASS_SUPPORT")
    if metrics.raw_ece_10_bin > config.maximum_ece:
        blockers.append("ECE_RELEASE_GATE_FAILED")
    if brier_skill <= config.minimum_brier_skill:
        blockers.append("BRIER_SKILL_RELEASE_GATE_FAILED")
    if (
        metrics.multiclass_brier > raw_metrics.multiclass_brier
        or metrics.log_loss > raw_metrics.log_loss
    ):
        blockers.append("CALIBRATION_DEGRADATION_GATE_FAILED")
    if len(supported_bins) < config.minimum_supported_probability_bins:
        blockers.append("INSUFFICIENT_DISPLAY_PROBABILITY_BIN_SUPPORT")
    if not slice_gates or any(not item.passed for item in slice_gates):
        blockers.append("CHRONOLOGICAL_SLICE_STABILITY_GATE_FAILED")
    return tuple(blockers)


def _calibration_id(
    rows: tuple[CalibrationObservation, ...],
    config: CalibrationConfig,
    created_at: datetime,
) -> str:
    identity = json.dumps(
        {
            "calibration_identity": CALIBRATION_IDENTITY,
            "prediction_ids": [item.prediction_id for item in rows],
            "config": config.to_contract(),
            "created_at": created_at.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, f"calibration-run:{hashlib.sha256(identity.encode()).hexdigest()}"))

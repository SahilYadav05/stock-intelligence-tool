"""Immutable contracts for model drift and live-signal release readiness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re


SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReleaseStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    release_id: str
    created_at: datetime
    model_version: str
    model_sha256: str
    calibration_version: str
    calibration_sha256: str
    feature_version: str
    feature_set_hash: str
    label_version: str
    label_definition_hash: str
    signal_policy_version: str
    evaluation_run_id: str
    calibration_ece: float
    positive_brier_skill: bool

    def __post_init__(self) -> None:
        _aware(self.created_at, "created_at")
        for name in (
            "model_sha256",
            "calibration_sha256",
            "feature_set_hash",
            "label_definition_hash",
        ):
            if not SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not 0.0 <= self.calibration_ece <= 1.0:
            raise ValueError("calibration_ece must be in [0, 1]")
        for name in (
            "release_id",
            "model_version",
            "calibration_version",
            "feature_version",
            "label_version",
            "signal_policy_version",
            "evaluation_run_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")

    def public_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "release_id": self.release_id,
            "created_at": _time(self.created_at),
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "calibration_version": self.calibration_version,
            "calibration_sha256": self.calibration_sha256,
            "feature_version": self.feature_version,
            "feature_set_hash": self.feature_set_hash,
            "label_version": self.label_version,
            "label_definition_hash": self.label_definition_hash,
            "signal_policy_version": self.signal_policy_version,
            "evaluation_run_id": self.evaluation_run_id,
            "calibration_ece": self.calibration_ece,
            "positive_brier_skill": self.positive_brier_skill,
        }


@dataclass(frozen=True, slots=True)
class DriftEvidence:
    reference_id: str
    evaluated_at: datetime
    reference_samples: int
    current_samples: int
    feature_psi: float
    probability_jsd: float
    feature_threshold: float = 0.20
    probability_threshold: float = 0.10

    def __post_init__(self) -> None:
        _aware(self.evaluated_at, "evaluated_at")
        if not self.reference_id:
            raise ValueError("reference_id is required")
        if self.reference_samples < 100 or self.current_samples < 100:
            raise ValueError("Drift evidence requires at least 100 reference and current samples")
        if min(self.feature_psi, self.probability_jsd) < 0:
            raise ValueError("Drift scores cannot be negative")

    @property
    def passed(self) -> bool:
        return self.feature_psi <= self.feature_threshold and self.probability_jsd <= self.probability_threshold


@dataclass(frozen=True, slots=True)
class ReleaseReadiness:
    evaluated_at: datetime
    status: ReleaseStatus
    signal_allowed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    release_id: str | None
    exact_snapshot_match: bool
    security_mode: str
    kill_switch_active: bool

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "evaluated_at": _time(self.evaluated_at),
            "status": self.status.value,
            "signal_allowed": self.signal_allowed,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "release_id": self.release_id,
            "exact_snapshot_match": self.exact_snapshot_match,
            "security_mode": self.security_mode,
            "kill_switch_active": self.kill_switch_active,
        }


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")

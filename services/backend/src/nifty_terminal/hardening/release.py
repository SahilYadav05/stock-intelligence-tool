"""Deterministic release manifest loading and fail-closed signal gates."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from nifty_terminal.dashboard.models import AnalysisView
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.hardening.models import (
    DriftEvidence,
    ReleaseManifest,
    ReleaseReadiness,
    ReleaseStatus,
)
from nifty_terminal.ml.definitions import LABEL_DEFINITION_HASH, LABEL_VERSION
from nifty_terminal.signals.definitions import SIGNAL_POLICY_VERSION
from nifty_terminal.settings import Settings


def load_release_manifest(path: Path | None) -> ReleaseManifest | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "release_id", "created_at", "model_version", "model_sha256",
        "calibration_version", "calibration_sha256", "feature_version",
        "feature_set_hash", "label_version", "label_definition_hash",
        "signal_policy_version", "evaluation_run_id", "calibration_ece",
        "positive_brier_skill",
    }
    if set(payload) != required:
        raise ValueError("Release manifest fields do not match artifact-manifest.v1")
    positive_brier_skill = payload["positive_brier_skill"]
    if not isinstance(positive_brier_skill, bool):
        raise ValueError("positive_brier_skill must be a boolean")
    return ReleaseManifest(
        release_id=_string(payload, "release_id"),
        created_at=datetime.fromisoformat(_string(payload, "created_at").replace("Z", "+00:00")),
        model_version=_string(payload, "model_version"),
        model_sha256=_string(payload, "model_sha256"),
        calibration_version=_string(payload, "calibration_version"),
        calibration_sha256=_string(payload, "calibration_sha256"),
        feature_version=_string(payload, "feature_version"),
        feature_set_hash=_string(payload, "feature_set_hash"),
        label_version=_string(payload, "label_version"),
        label_definition_hash=_string(payload, "label_definition_hash"),
        signal_policy_version=_string(payload, "signal_policy_version"),
        evaluation_run_id=_string(payload, "evaluation_run_id"),
        calibration_ece=float(payload["calibration_ece"]),
        positive_brier_skill=positive_brier_skill,
    )


def evaluate_release(
    *,
    evaluated_at: datetime,
    settings: Settings,
    manifest: ReleaseManifest | None,
    market_view: MarketStateView | None,
    analysis: AnalysisView | None,
    drift: DriftEvidence | None,
) -> ReleaseReadiness:
    blockers: list[str] = []
    warnings: list[str] = []
    exact_snapshot_match = bool(
        market_view
        and analysis
        and analysis.snapshot_id == market_view.snapshot.snapshot_id
        and analysis.candle_revision_checksum == market_view.snapshot.candle_revision_checksum
    )

    if settings.market_data_mode != "live" or not settings.market_data_provider:
        blockers.append("LICENSED_LIVE_PROVIDER_NOT_CONFIGURED")
    if market_view is None or market_view.snapshot.data_status is not ConnectionState.LIVE:
        blockers.append("CANONICAL_MARKET_DATA_NOT_LIVE")
    if not exact_snapshot_match:
        blockers.append("CHART_MODEL_SNAPSHOT_NOT_SYNCHRONIZED")
    if manifest is None:
        blockers.append("APPROVED_RELEASE_MANIFEST_MISSING")
    else:
        if not _artifact_matches(settings.model_artifact_path, manifest.model_sha256):
            blockers.append("MODEL_ARTIFACT_MISSING_OR_HASH_MISMATCH")
        if not _artifact_matches(
            settings.calibration_artifact_path, manifest.calibration_sha256
        ):
            blockers.append("CALIBRATION_ARTIFACT_MISSING_OR_HASH_MISMATCH")
        if manifest.feature_version != FEATURE_VERSION or manifest.feature_set_hash != FEATURE_SET_HASH:
            blockers.append("FEATURE_ARTIFACT_INCOMPATIBLE")
        if manifest.label_version != LABEL_VERSION or manifest.label_definition_hash != LABEL_DEFINITION_HASH:
            blockers.append("LABEL_ARTIFACT_INCOMPATIBLE")
        if manifest.signal_policy_version != SIGNAL_POLICY_VERSION:
            blockers.append("SIGNAL_POLICY_ARTIFACT_INCOMPATIBLE")
        if manifest.calibration_ece > 0.05 or not manifest.positive_brier_skill:
            blockers.append("CALIBRATION_RELEASE_GATE_FAILED")
        if analysis and (
            analysis.model_version != manifest.model_version
            or analysis.calibration_version != manifest.calibration_version
            or analysis.feature_version != manifest.feature_version
        ):
            blockers.append("LIVE_ANALYSIS_ARTIFACT_MISMATCH")
    if drift is None:
        blockers.append("DRIFT_REFERENCE_EVIDENCE_MISSING")
    elif not drift.passed:
        blockers.append("DRIFT_THRESHOLD_BREACHED")
    if settings.live_signal_kill_switch:
        blockers.append("LIVE_SIGNAL_KILL_SWITCH_ACTIVE")
    if settings.environment != "production":
        warnings.append("NON_PRODUCTION_RUNTIME")
    if settings.api_auth_mode != "bearer":
        warnings.append("API_BEARER_AUTH_DISABLED")

    unique_blockers = tuple(dict.fromkeys(blockers))
    return ReleaseReadiness(
        evaluated_at=evaluated_at,
        status=ReleaseStatus.BLOCKED if unique_blockers else ReleaseStatus.READY,
        signal_allowed=not unique_blockers,
        blockers=unique_blockers,
        warnings=tuple(dict.fromkeys(warnings)),
        release_id=manifest.release_id if manifest else None,
        exact_snapshot_match=exact_snapshot_match,
        security_mode=settings.api_auth_mode.upper(),
        kill_switch_active=settings.live_signal_kill_switch,
    )


def _artifact_matches(path: Path | None, expected_hash: str) -> bool:
    if path is None or not path.is_file():
        return False
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected_hash


def _string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value

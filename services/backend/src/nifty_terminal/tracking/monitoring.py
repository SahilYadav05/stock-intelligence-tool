"""Fail-safe health view for the live market, model, and tracking pipeline."""

from __future__ import annotations

from datetime import datetime

from nifty_terminal.dashboard.models import AnalysisView
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.hardening.models import DriftEvidence
from nifty_terminal.tracking.models import (
    EvidenceStatus,
    MonitorStatus,
    MonitoringCheck,
    MonitoringView,
    PredictionAnalytics,
)


def build_monitoring_view(
    *,
    instrument_id: str,
    generated_at: datetime,
    market_view: MarketStateView | None,
    analysis: AnalysisView | None,
    analytics: PredictionAnalytics,
    drift_evidence: DriftEvidence | None = None,
) -> MonitoringView:
    checks: list[MonitoringCheck] = []
    if market_view is None:
        checks.extend(
            (
                _check("MARKET_DATA", MonitorStatus.CRITICAL, generated_at, "No canonical market snapshot"),
                _check("DATA_FRESHNESS", MonitorStatus.UNAVAILABLE, generated_at, "Data age unavailable"),
                _check("SNAPSHOT_SYNC", MonitorStatus.UNAVAILABLE, generated_at, "No chart snapshot to verify"),
            )
        )
    else:
        state = market_view.snapshot.data_status
        market_status = MonitorStatus.OK if state is ConnectionState.LIVE else MonitorStatus.CRITICAL
        checks.append(_check("MARKET_DATA", market_status, generated_at, state.value))
        freshness_status = MonitorStatus.OK if state in {ConnectionState.LIVE, ConnectionState.MARKET_CLOSED} else MonitorStatus.WARN
        checks.append(
            _check(
                "DATA_FRESHNESS",
                freshness_status,
                generated_at,
                f"Data as of {market_view.snapshot.data_as_of.isoformat()}",
            )
        )
        synced = bool(
            analysis
            and analysis.snapshot_id == market_view.snapshot.snapshot_id
            and analysis.candle_revision_checksum
            == market_view.snapshot.candle_revision_checksum
        )
        checks.append(
            _check(
                "SNAPSHOT_SYNC",
                MonitorStatus.OK if synced else MonitorStatus.CRITICAL,
                generated_at,
                "Chart and analysis revisions match" if synced else "Exact analysis snapshot unavailable",
            )
        )

    checks.append(
        _check(
            "MODEL_RELEASE",
            MonitorStatus.OK if analysis else MonitorStatus.UNAVAILABLE,
            generated_at,
            analysis.model_version if analysis else "No approved live model artifact",
        )
    )
    checks.append(
        _check(
            "CALIBRATION",
            MonitorStatus.OK if analysis else MonitorStatus.UNAVAILABLE,
            generated_at,
            analysis.calibration_version if analysis else "No approved calibration artifact",
        )
    )
    coverage_status = (
        MonitorStatus.UNAVAILABLE
        if analytics.tracked_predictions == 0
        else MonitorStatus.OK
        if analytics.pending_predictions == 0
        else MonitorStatus.WARN
    )
    checks.append(
        _check(
            "OUTCOME_TRACKING",
            coverage_status,
            generated_at,
            f"{analytics.assessed_predictions}/{analytics.tracked_predictions} predictions assessed",
        )
    )
    if drift_evidence is None:
        model_drift = EvidenceStatus.UNAVAILABLE
        probability_drift = EvidenceStatus.UNAVAILABLE
        checks.append(
            _check(
                "DRIFT_BASELINE",
                MonitorStatus.UNAVAILABLE,
                generated_at,
                "No versioned reference distribution; drift cannot be claimed ready",
            )
        )
    else:
        model_drift = (
            EvidenceStatus.READY
            if drift_evidence.feature_psi <= drift_evidence.feature_threshold
            else EvidenceStatus.BREACHED
        )
        probability_drift = (
            EvidenceStatus.READY
            if drift_evidence.probability_jsd <= drift_evidence.probability_threshold
            else EvidenceStatus.BREACHED
        )
    statuses = {item.status for item in checks}
    if EvidenceStatus.BREACHED in {model_drift, probability_drift}:
        statuses.add(MonitorStatus.CRITICAL)
    overall = (
        MonitorStatus.CRITICAL
        if MonitorStatus.CRITICAL in statuses
        else MonitorStatus.WARN
        if MonitorStatus.WARN in statuses
        else MonitorStatus.OK
        if MonitorStatus.OK in statuses
        else MonitorStatus.UNAVAILABLE
    )
    return MonitoringView(
        instrument_id=instrument_id,
        generated_at=generated_at,
        overall_status=overall,
        checks=tuple(checks),
        model_drift_status=model_drift,
        probability_drift_status=probability_drift,
        alerting_enabled=False,
    )


def _check(key: str, status: MonitorStatus, observed_at: datetime, detail: str) -> MonitoringCheck:
    return MonitoringCheck(key=key, status=status, observed_at=observed_at, detail=detail)

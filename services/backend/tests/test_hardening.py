from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from dashboard_fixture import build_analysis_view
from market_state_fixture import build_market_state_view
from nifty_terminal.api.app import create_app
from nifty_terminal.dashboard.read_model import InMemoryAnalysisReadModel
from nifty_terminal.delivery.read_model import InMemoryMarketStateReadModel
from nifty_terminal.delivery.service import MarketStateDeliveryService
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.hardening.audit import SQLiteAuditLedger
from nifty_terminal.hardening.circuit_breaker import SignalCircuitBreaker
from nifty_terminal.hardening.drift import jensen_shannon_divergence, population_stability_index
from nifty_terminal.hardening.models import DriftEvidence, ReleaseManifest, ReleaseStatus
from nifty_terminal.hardening.release import evaluate_release
from nifty_terminal.ml.definitions import LABEL_DEFINITION_HASH, LABEL_VERSION
from nifty_terminal.settings import Settings
from nifty_terminal.signals.definitions import SIGNAL_POLICY_VERSION


NOW = datetime(2026, 8, 24, 10, 35, tzinfo=timezone.utc)


def production_like_settings(**overrides) -> Settings:
    values = {
        "app_name": "NIFTY Test",
        "environment": "production",
        "log_level": "WARNING",
        "market_data_mode": "live",
        "market_data_provider": "licensed-provider",
        "api_allowed_origins": ("https://terminal.example",),
        "api_auth_mode": "bearer",
        "api_auth_token": "a" * 40,
        "live_signal_kill_switch": False,
    }
    values.update(overrides)
    return Settings(**values)


def manifest_for_fixture(
    *,
    model_sha256: str = "1" * 64,
    calibration_sha256: str = "2" * 64,
) -> ReleaseManifest:
    return ReleaseManifest(
        release_id="release-fixture-v1",
        created_at=NOW,
        model_version="research-model-fixture",
        model_sha256=model_sha256,
        calibration_version="multiclass_temperature.v1",
        calibration_sha256=calibration_sha256,
        feature_version=FEATURE_VERSION,
        feature_set_hash=FEATURE_SET_HASH,
        label_version=LABEL_VERSION,
        label_definition_hash=LABEL_DEFINITION_HASH,
        signal_policy_version=SIGNAL_POLICY_VERSION,
        evaluation_run_id="walk-forward-fixture",
        calibration_ece=0.03,
        positive_brier_skill=True,
    )


def drift_evidence(*, feature_psi: float = 0.05, probability_jsd: float = 0.02) -> DriftEvidence:
    return DriftEvidence(
        reference_id="reference-distribution-v1",
        evaluated_at=NOW,
        reference_samples=500,
        current_samples=150,
        feature_psi=feature_psi,
        probability_jsd=probability_jsd,
    )


class HardeningTests(TestCase):
    def test_release_gate_passes_only_with_every_evidence_layer(self) -> None:
        market = build_market_state_view()
        with TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.bin"
            calibration_path = Path(temporary) / "calibration.bin"
            model_path.write_bytes(b"verified model fixture")
            calibration_path.write_bytes(b"verified calibration fixture")
            readiness = evaluate_release(
                evaluated_at=NOW,
                settings=production_like_settings(
                    model_artifact_path=model_path,
                    calibration_artifact_path=calibration_path,
                ),
                manifest=manifest_for_fixture(
                    model_sha256=sha256(model_path.read_bytes()).hexdigest(),
                    calibration_sha256=sha256(calibration_path.read_bytes()).hexdigest(),
                ),
                market_view=market,
                analysis=build_analysis_view(market),
                drift=drift_evidence(),
            )
        self.assertEqual(readiness.status, ReleaseStatus.READY)
        self.assertTrue(readiness.signal_allowed)
        self.assertEqual(readiness.blockers, ())

    def test_missing_drift_and_kill_switch_fail_closed(self) -> None:
        market = build_market_state_view()
        readiness = evaluate_release(
            evaluated_at=NOW,
            settings=production_like_settings(live_signal_kill_switch=True),
            manifest=manifest_for_fixture(),
            market_view=market,
            analysis=build_analysis_view(market),
            drift=None,
        )
        self.assertFalse(readiness.signal_allowed)
        self.assertIn("DRIFT_REFERENCE_EVIDENCE_MISSING", readiness.blockers)
        self.assertIn("LIVE_SIGNAL_KILL_SWITCH_ACTIVE", readiness.blockers)
        self.assertIn("MODEL_ARTIFACT_MISSING_OR_HASH_MISMATCH", readiness.blockers)

    def test_drift_breach_blocks_release(self) -> None:
        market = build_market_state_view()
        readiness = evaluate_release(
            evaluated_at=NOW,
            settings=production_like_settings(),
            manifest=manifest_for_fixture(),
            market_view=market,
            analysis=build_analysis_view(market),
            drift=drift_evidence(feature_psi=0.25),
        )
        self.assertIn("DRIFT_THRESHOLD_BREACHED", readiness.blockers)

    def test_drift_metrics_are_zero_for_identical_distributions(self) -> None:
        distribution = (10.0, 20.0, 30.0)
        self.assertAlmostEqual(population_stability_index(distribution, distribution), 0.0)
        self.assertAlmostEqual(jensen_shannon_divergence(distribution, distribution), 0.0)

    def test_api_authentication_headers_and_readiness_are_fail_closed(self) -> None:
        application = create_app(settings=production_like_settings())
        with TestClient(application) as client:
            public = client.get("/api/v1/live")
            rejected = client.get("/api/v1/health")
            accepted = client.get("/api/v1/health", headers={"Authorization": f"Bearer {'a' * 40}"})
            ready = client.get("/api/v1/ready", headers={"Authorization": f"Bearer {'a' * 40}"})
        self.assertEqual(public.status_code, 200)
        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(ready.status_code, 503)
        self.assertFalse(ready.json()["detail"]["signal_allowed"])
        for response in (public, rejected, accepted, ready):
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            self.assertIn("max-age=31536000", response.headers["strict-transport-security"])

    def test_oversized_request_is_rejected_before_routing(self) -> None:
        settings = production_like_settings(request_body_limit_bytes=1024)
        application = create_app(settings=settings)
        with TestClient(application) as client:
            response = client.get(
                "/api/v1/health",
                headers={"Content-Length": "2048", "Authorization": f"Bearer {'a' * 40}"},
            )
        self.assertEqual(response.status_code, 413)

    def test_exact_origin_preflight_is_not_blocked_by_bearer_auth(self) -> None:
        application = create_app(settings=production_like_settings())
        with TestClient(application) as client:
            response = client.options(
                "/api/v1/health",
                headers={
                    "Origin": "https://terminal.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://terminal.example")

    def test_production_environment_validation_rejects_insecure_defaults(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "production",
                "API_AUTH_MODE": "disabled",
                "API_ALLOWED_ORIGINS": "http://localhost:5173",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Production requires"):
                Settings.from_environment()

    def test_circuit_breaker_requires_passed_release_gate_to_reset(self) -> None:
        breaker = SignalCircuitBreaker()
        with self.assertRaisesRegex(ValueError, "release gates"):
            breaker.reset(changed_at=NOW, release_gate_passed=False)
        self.assertFalse(breaker.reset(changed_at=NOW, release_gate_passed=True).open)

    def test_audit_ledger_is_hash_chained_idempotent_and_append_only(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.sqlite3"
            ledger = SQLiteAuditLedger(path)
            inserted = ledger.append(
                event_id="audit-1",
                occurred_at=NOW,
                category="RELEASE",
                action="GATE_EVALUATED",
                actor="system",
                details={"status": "BLOCKED"},
            )
            duplicate = ledger.append(
                event_id="audit-1",
                occurred_at=NOW,
                category="RELEASE",
                action="GATE_EVALUATED",
                actor="system",
                details={"status": "BLOCKED"},
            )
            self.assertTrue(inserted)
            self.assertFalse(duplicate)
            self.assertTrue(ledger.verify_chain())
            with closing(sqlite3.connect(path)) as connection:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("DELETE FROM security_audit_events")

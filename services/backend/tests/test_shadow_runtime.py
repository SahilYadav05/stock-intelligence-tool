from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from history_fixture import SESSION_OPEN
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.models import FeatureSnapshot
from nifty_terminal.research.step16 import CalibrationArtifactV2, build_shadow_artifact
from nifty_terminal.research.step17 import (
    ShadowPolicyThresholds,
    build_policy_artifact,
    policy_direction,
)
from nifty_terminal.shadow.artifacts import load_shadow_artifacts
from nifty_terminal.shadow.ledger import SQLiteShadowLedger
from nifty_terminal.shadow.runtime import _assess_price_action_path, _first_touch_outcome

from test_locked_shadow_research import _dataset_report
from test_ml_labels import _minute_candles


class ShadowRuntimeTests(TestCase):
    def test_policy_uses_prior_lift_and_can_remain_wait_only(self) -> None:
        thresholds = ShadowPolicyThresholds(
            "RAW_MODEL_SCORE", 0.40, 0.04, 0.45, 0.03
        )
        prior = np.asarray([0.40, 0.20, 0.40])
        self.assertIsNone(
            policy_direction(np.asarray([0.42, 0.16, 0.42]), prior, thresholds)
        )
        self.assertEqual(
            policy_direction(np.asarray([0.20, 0.20, 0.60]), prior, thresholds),
            "BUY",
        )

    def test_safe_json_artifacts_load_and_reproduce_probability_vectors(self) -> None:
        model = build_shadow_artifact(
            dataset=_dataset_report(),
            calibrator=CalibrationArtifactV2("identity", {}),
        )
        policy = build_policy_artifact(
            dataset_id="dataset-step16",
            thresholds=ShadowPolicyThresholds(
                "RAW_MODEL_SCORE", 0.40, 0.02, 0.45, 0.00
            ),
            deployment_prior=np.asarray([1 / 3, 1 / 3, 1 / 3]),
            historical_policy_gate_passed=False,
            blockers=("FORWARD_CONFIRMATION_NOT_COMPLETED",),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.json"
            policy_path = root / "policy.json"
            model_path.write_text(json.dumps(model), encoding="utf-8")
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "manifest_version": "nifty_shadow_runtime.v1",
                "dataset_id": "dataset-step16",
                "model_artifact_path": str(model_path),
                "model_artifact_sha256": model["sha256"],
                "policy_artifact_path": str(policy_path),
                "policy_artifact_sha256": policy["sha256"],
                "prediction_collection_enabled": True,
                "shadow_candidate_directions_enabled": False,
                "runtime_mode": "WAIT_ONLY",
                "official_signal_available": False,
                "precise_probability_display_allowed": False,
                "automatic_trading_enabled": False,
                "live_signal_kill_switch_required": True,
            }
            checksum = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest["sha256"] = checksum
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            loaded = load_shadow_artifacts(manifest_path)
            raw, calibrated = loaded.predict(
                FeatureSnapshot(
                    schema_version=1,
                    feature_snapshot_id="feature-snapshot",
                    feature_version=FEATURE_VERSION,
                    feature_set_hash=FEATURE_SET_HASH,
                    market_snapshot_id="market-snapshot",
                    instrument_id="NIFTY50_SPOT",
                    decision_time=SESSION_OPEN,
                    input_revision_checksum="checksum",
                    values=(("feature_a", Decimal("3")), ("feature_b", Decimal("2"))),
                    is_ready=True,
                    blockers=(),
                )
            )

            self.assertAlmostEqual(float(raw.sum()), 1.0)
            np.testing.assert_allclose(raw, calibrated)
            self.assertEqual(loaded.shadow_direction(raw, calibrated), "WAIT")

    def test_shadow_ledger_is_idempotent_and_append_only(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "shadow.sqlite3"
            ledger = SQLiteShadowLedger(path)
            prediction = {
                "prediction_id": "prediction-1",
                "snapshot_id": "snapshot-1",
                "decision_time": SESSION_OPEN.isoformat(),
                "outcome_due_at": (SESSION_OPEN + timedelta(minutes=60)).isoformat(),
            }
            self.assertTrue(ledger.append_prediction(prediction))
            self.assertFalse(ledger.append_prediction(prediction))
            with self.assertRaises(ValueError):
                ledger.append_prediction({**prediction, "snapshot_id": "changed"})
            self.assertEqual(
                len(ledger.pending(due_at_or_before=SESSION_OPEN + timedelta(minutes=60))),
                1,
            )
            assessment = {
                "assessment_id": "assessment-1",
                "prediction_id": "prediction-1",
                "assessed_at": (SESSION_OPEN + timedelta(minutes=60)).isoformat(),
            }
            self.assertTrue(ledger.append_assessment(assessment))
            self.assertEqual(ledger.status()["assessment_count"], 1)
            with closing(sqlite3.connect(path)) as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE shadow_predictions SET snapshot_id = 'x' WHERE prediction_id = 'prediction-1'"
                    )

    def test_forward_outcome_uses_one_minute_first_touch_and_marks_ambiguity(self) -> None:
        candles = list(_minute_candles(SESSION_OPEN, 60))
        candles[0] = replace(
            candles[0], high=Decimal("103"), low=Decimal("97")
        )
        outcome, first_touch = _first_touch_outcome(
            tuple(candles), up=Decimal("102"), down=Decimal("98")
        )
        self.assertEqual(outcome, "AMBIGUOUS")
        self.assertEqual(first_touch, candles[0].closes_at)

    def test_shadow_contracts_cannot_enable_official_signals(self) -> None:
        root = Path(__file__).resolve().parents[3]
        for filename in (
            "shadow-policy-research.v1.schema.json",
            "shadow-runtime-manifest.v1.schema.json",
        ):
            schema = json.loads((root / "contracts" / filename).read_text(encoding="utf-8"))
            self.assertFalse(schema["properties"]["official_signal_available"]["const"])

    def test_price_action_shadow_path_tracks_multiple_targets_conservatively(self) -> None:
        candles = list(_minute_candles(SESSION_OPEN, 60))
        candles[1] = replace(candles[1], high=Decimal("104.5"))
        prediction = {
            "price_action_analysis": {
                "trade_plan": {
                    "direction": "BUY",
                    "trigger": "100",
                    "stop": "98",
                    "target1": "102",
                    "target2": "104",
                    "target3": "106",
                }
            }
        }

        result = _assess_price_action_path(prediction, tuple(candles))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "EXPIRED")
        self.assertEqual(result["maximum_target_reached"], 2)
        self.assertFalse(result["stop_hit"])

        candles[0] = replace(candles[0], high=Decimal("103"), low=Decimal("97"))
        stopped = _assess_price_action_path(prediction, tuple(candles))
        assert stopped is not None
        self.assertEqual(stopped["status"], "STOPPED")
        self.assertEqual(stopped["maximum_target_reached"], 0)

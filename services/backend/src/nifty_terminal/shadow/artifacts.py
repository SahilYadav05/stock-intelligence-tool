"""Hash-verified safe-JSON shadow artifact loading and inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.models import FeatureSnapshot
from nifty_terminal.ml.definitions import CLASS_ORDER
from nifty_terminal.research.step16 import CalibrationArtifactV2, apply_calibrator
from nifty_terminal.research.step17 import ShadowPolicyThresholds, policy_direction


@dataclass(frozen=True, slots=True)
class LoadedShadowArtifacts:
    manifest_path: Path
    manifest_sha256: str
    model: dict[str, object]
    policy: dict[str, object]

    def predict(self, features: FeatureSnapshot) -> tuple[np.ndarray, np.ndarray]:
        if not features.is_ready:
            raise ValueError("Shadow inference requires a ready feature snapshot")
        if features.feature_version != self.model["feature_version"]:
            raise ValueError("Shadow model feature version mismatch")
        if features.feature_set_hash != self.model["feature_set_hash"]:
            raise ValueError("Shadow model feature-set hash mismatch")
        names = tuple(str(item) for item in self.model["feature_names"])
        values = dict(features.values)
        if set(values) != set(names):
            raise ValueError("Shadow feature names do not exactly match the artifact")
        ordered = []
        for name in names:
            value = values[name]
            if value is None:
                raise ValueError(f"Shadow feature is unavailable: {name}")
            ordered.append(float(value))
        x = np.asarray(ordered, dtype=float)
        mean = np.asarray(self.model["scaler_mean"], dtype=float)
        scale = np.asarray(self.model["scaler_scale"], dtype=float)
        if x.shape != mean.shape or mean.shape != scale.shape or np.any(scale <= 0):
            raise ValueError("Shadow scaler parameters are invalid")
        standardized = (x - mean) / scale
        coefficients = np.asarray(self.model["coefficients"], dtype=float)
        intercepts = np.asarray(self.model["intercepts"], dtype=float)
        classes = tuple(str(item) for item in self.model["classes"])
        logits = standardized @ coefficients.T + intercepts
        logits -= logits.max()
        raw = np.exp(logits)
        raw /= raw.sum()
        raw = raw[[classes.index(name) for name in CLASS_ORDER]]
        calibration_contract = self.model["calibration"]
        calibrator = CalibrationArtifactV2(
            method=str(calibration_contract["method"]),
            parameters=dict(calibration_contract["parameters"]),
        )
        calibrated = apply_calibrator(calibrator, raw.reshape(1, -1))[0]
        return raw, calibrated

    def shadow_direction(self, raw: np.ndarray, calibrated: np.ndarray) -> str:
        if not self.policy["shadow_candidate_directions_enabled"]:
            return "WAIT"
        thresholds_contract = self.policy["thresholds"]
        thresholds = ShadowPolicyThresholds(
            score_source=str(thresholds_contract["score_source"]),
            activation_score=float(thresholds_contract["activation_score"]),
            minimum_class_margin=float(thresholds_contract["minimum_class_margin"]),
            maximum_neither_score=float(thresholds_contract["maximum_neither_score"]),
            minimum_prior_lift=float(thresholds_contract["minimum_prior_lift"]),
        )
        prior_contract = self.policy["deployment_prior"]
        prior = np.asarray(
            [prior_contract["DOWN"], prior_contract["NEITHER"], prior_contract["UP"]],
            dtype=float,
        )
        scores = raw if thresholds.score_source == "RAW_MODEL_SCORE" else calibrated
        return policy_direction(scores, prior, thresholds) or "WAIT"


def load_shadow_artifacts(manifest_path: Path) -> LoadedShadowArtifacts:
    manifest = _load_verified(manifest_path)
    if manifest.get("manifest_version") != "nifty_shadow_runtime.v1":
        raise ValueError("Unsupported shadow runtime manifest")
    if manifest.get("official_signal_available") is not False:
        raise ValueError("Shadow manifest must prohibit official signals")
    if manifest.get("automatic_trading_enabled") is not False:
        raise ValueError("Shadow manifest must prohibit automatic trading")
    model_path = Path(str(manifest["model_artifact_path"]))
    policy_path = Path(str(manifest["policy_artifact_path"]))
    model = _load_verified(model_path, str(manifest["model_artifact_sha256"]))
    policy = _load_verified(policy_path, str(manifest["policy_artifact_sha256"]))
    if model.get("shadow_only") is not True or model.get("approved_for_live_inference") is not False:
        raise ValueError("Model artifact is not shadow-only")
    if policy.get("shadow_only") is not True or policy.get("approved_for_live_inference") is not False:
        raise ValueError("Policy artifact is not shadow-only")
    if model.get("dataset_id") != policy.get("dataset_id") or model.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("Shadow artifacts do not share one dataset identity")
    if model.get("feature_version") != FEATURE_VERSION or model.get("feature_set_hash") != FEATURE_SET_HASH:
        raise ValueError("Shadow model is incompatible with the live feature engine")
    if set(str(item) for item in model.get("classes", [])) != set(CLASS_ORDER):
        raise ValueError("Shadow model class order is incomplete")
    return LoadedShadowArtifacts(
        manifest_path=manifest_path,
        manifest_sha256=str(manifest["sha256"]),
        model=model,
        policy=policy,
    )


def _load_verified(path: Path, expected: str | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact root must be an object: {path}")
    embedded = str(payload.get("sha256", ""))
    if expected is not None and embedded != expected:
        raise ValueError(f"Artifact does not match manifest checksum: {path}")
    body = dict(payload)
    body.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if embedded != actual:
        raise ValueError(f"Artifact checksum verification failed: {path}")
    return payload

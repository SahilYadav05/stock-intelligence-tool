"""Versioned Step 7 calibration and release-gate definitions."""

from __future__ import annotations

import hashlib
import json


CALIBRATION_VERSION = "multiclass_temperature.v1"
RELEASE_GATE_VERSION = "calibration_release_gate.v1"

DEFAULT_CALIBRATION_CONFIG = {
    "fit_fraction": 0.60,
    "minimum_total_predictions": 500,
    "minimum_fit_class_support": 50,
    "minimum_evaluation_class_support": 30,
    "minimum_supported_probability_bin": 30,
    "minimum_supported_probability_bins": 2,
    "maximum_ece": 0.05,
    "minimum_brier_skill": 0.0,
    "maximum_slice_ece": 0.10,
    "minimum_slice_brier_skill": -0.10,
    "minimum_slice_samples": 50,
}

CALIBRATION_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "calibration_version": CALIBRATION_VERSION,
            "release_gate_version": RELEASE_GATE_VERSION,
            "defaults": DEFAULT_CALIBRATION_CONFIG,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

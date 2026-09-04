"""Versioned research defaults for Step 7 signal and risk policy."""

from __future__ import annotations

import hashlib
import json


SIGNAL_POLICY_VERSION = "wait_first_atr_policy.v1"
RISK_POLICY_VERSION = "atr_levels.v1"
LIFECYCLE_POLICY_VERSION = "immutable_signal_lifecycle.v1"

# These are conservative research defaults, not statistically approved constants.
DEFAULT_POLICY_DEFINITION = {
    "activation_probability": 0.60,
    "minimum_directional_margin": 0.15,
    "maximum_neither_probability": 0.45,
    "minimum_expected_atr": 0.15,
    "entry_half_width_atr": 0.10,
    "stop_atr": 0.75,
    "target_atr": [1.0, 1.5, 2.0],
    "minimum_target1_reward_risk": 1.25,
    "reversal_probability": 0.72,
    "reversal_margin": 0.25,
    "expiry_bars": 12,
}

SIGNAL_POLICY_IDENTITY = hashlib.sha256(
    json.dumps(DEFAULT_POLICY_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

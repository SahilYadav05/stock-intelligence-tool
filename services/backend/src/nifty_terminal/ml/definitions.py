"""Immutable Step 6 target, validation, and research-model definitions."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json

from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION


LABEL_VERSION = "nifty_5m_atr_first_touch.v1"
RESEARCH_VERSION = "ml_research.v1"
PRIMARY_TIMEFRAME_MINUTES = 5
HORIZON_BARS = 12
HORIZON_MINUTES = PRIMARY_TIMEFRAME_MINUTES * HORIZON_BARS
UP_ATR_MULTIPLIER = Decimal("1.0")
DOWN_ATR_MULTIPLIER = Decimal("1.0")
DEFAULT_PURGE_BARS = HORIZON_BARS
DEFAULT_EMBARGO_BARS = HORIZON_BARS
CLASS_ORDER = ("DOWN", "NEITHER", "UP")
RANDOM_SEED = 1701

MODEL_CANDIDATE_DEFINITIONS = {
    "multinomial_logistic": {
        "preprocessing": "StandardScaler fitted inside each training fold",
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_seed": RANDOM_SEED,
    },
    "hist_gradient_boosting": {
        "class_weight": "balanced",
        "early_stopping": False,
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
        "max_iter": 160,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "random_seed": RANDOM_SEED,
    },
}

LABEL_DEFINITION = {
    "reference": "finalized 5m close at decision_time",
    "volatility": "Wilder ATR(14) from finalized 5m candles at decision_time",
    "up": "+1.0 ATR touched first",
    "down": "-1.0 ATR touched first",
    "neither": "neither barrier touched in 12 subsequent finalized 5m candles",
    "ambiguous": "both barriers touched before lower-resolution ordering can resolve",
    "session": "the complete 60-minute outcome window must remain inside one NSE session",
}

LABEL_DEFINITION_HASH = hashlib.sha256(
    json.dumps(LABEL_DEFINITION, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "research_version": RESEARCH_VERSION,
            "label_version": LABEL_VERSION,
            "label_definition_hash": LABEL_DEFINITION_HASH,
            "feature_version": FEATURE_VERSION,
            "feature_set_hash": FEATURE_SET_HASH,
            "class_order": CLASS_ORDER,
            "model_candidate_definitions": MODEL_CANDIDATE_DEFINITIONS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

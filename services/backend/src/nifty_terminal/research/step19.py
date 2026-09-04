"""Step 19: baseline-controlled research with causal market-structure features."""

from __future__ import annotations

import hashlib
import json

from nifty_terminal.features.research_v4 import (
    PRICE_ACTION_FEATURE_SET_HASH,
    PRICE_ACTION_RESEARCH_VERSION,
)
from nifty_terminal.research.step18f import RESEARCH_IDENTITY as STEP18F_IDENTITY
from nifty_terminal.research.step18f import run_baseline_controlled_research


STEP19_VERSION = "price_action_baseline_controlled_research.v1"
RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP19_VERSION,
            "parent_methodology": STEP18F_IDENTITY,
            "price_action_feature_version": PRICE_ACTION_RESEARCH_VERSION,
            "price_action_feature_hash": PRICE_ACTION_FEATURE_SET_HASH,
            "change": "feature information only; gates and execution remain locked",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def run_price_action_research(*, context, minute_candles, context_bundle_sha256: str):
    payload = run_baseline_controlled_research(
        context=context,
        minute_candles=minute_candles,
        context_bundle_sha256=context_bundle_sha256,
    )
    payload["step19_version"] = STEP19_VERSION
    payload["parent_step18f_research_identity"] = payload["research_identity"]
    payload["research_identity"] = RESEARCH_IDENTITY
    payload["price_action_features"] = {
        "version": PRICE_ACTION_RESEARCH_VERSION,
        "feature_set_hash": PRICE_ACTION_FEATURE_SET_HASH,
        "causal_confirmed_pivots": True,
        "future_confirmed_pivots_used": False,
        "changes_to_locked_gates": False,
        "changes_to_execution_simulation": False,
    }
    payload["known_limitations_not_silently_filled"] = sorted(
        set(payload["known_limitations_not_silently_filled"])
        | {"PRICE_ACTION_FEATURES_REQUIRE_GENUINELY_FUTURE_CONFIRMATION"}
    )
    return payload

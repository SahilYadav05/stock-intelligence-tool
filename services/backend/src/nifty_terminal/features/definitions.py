"""Versioned definitions for price-only MVP features."""

from __future__ import annotations

import hashlib
import json


FEATURE_VERSION = "price_features.v1"
MINIMUM_HISTORY = 50

# Formula text is intentionally part of the immutable feature-set identity.
FEATURE_DEFINITIONS = (
    ("return_1", "close_t / close_t-1 - 1"),
    ("return_5", "close_t / close_t-5 - 1"),
    ("log_return_1", "ln(close_t / close_t-1)"),
    ("range_pct", "(high_t - low_t) / close_t"),
    ("body_pct", "(close_t - open_t) / open_t"),
    ("upper_wick_pct", "(high_t - max(open_t, close_t)) / open_t"),
    ("lower_wick_pct", "(min(open_t, close_t) - low_t) / open_t"),
    ("sma_20", "mean(close_t-19..t)"),
    ("sma_50", "mean(close_t-49..t)"),
    ("ema_20", "EMA close alpha=2/(20+1), SMA seed"),
    ("ema_50", "EMA close alpha=2/(50+1), SMA seed"),
    ("atr_14", "Wilder ATR(14), arithmetic TR seed"),
    ("atr_pct", "ATR14_t / close_t"),
    ("rsi_14", "Wilder RSI(14)"),
    ("rolling_vol_20", "population_std(return_1 over last 20 observations)"),
    ("bollinger_z_20", "(close_t - SMA20_t) / population_std(close last 20)"),
    ("roc_5", "close_t / close_t-5 - 1"),
    ("roc_12", "close_t / close_t-12 - 1"),
    ("distance_ema20_atr", "(close_t - EMA20_t) / ATR14_t"),
    ("range_atr", "(high_t - low_t) / ATR14_t"),
    ("trend_ema20_above_ema50", "EMA20_t > EMA50_t"),
    ("breakout_up_20", "close_t > max(high_t-20..t-1)"),
    ("breakout_down_20", "close_t < min(low_t-20..t-1)"),
    ("minute_of_session", "minutes from explicit NSE session open"),
    ("minutes_to_session_close", "minutes from candle close to explicit NSE close"),
    ("day_of_week", "exchange-local weekday Monday=0"),
)

FEATURE_SET_HASH = hashlib.sha256(
    json.dumps(FEATURE_DEFINITIONS, separators=(",", ":")).encode("utf-8")
).hexdigest()

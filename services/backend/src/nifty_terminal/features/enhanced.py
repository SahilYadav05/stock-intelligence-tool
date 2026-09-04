"""Causal cross-timeframe feature interactions shared by history and live inference."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

from nifty_terminal.ml.models import DatasetBuildReport, TrainingSample


ENHANCED_FEATURE_VERSION = "price_features.v2"
ENHANCED_FEATURE_DEFINITIONS = (
    ("trend_strength_5m", "(EMA20_5m - EMA50_5m) / ATR14_5m"),
    ("trend_strength_15m", "(EMA20_15m - EMA50_15m) / ATR14_15m"),
    ("trend_strength_1h", "(EMA20_1h - EMA50_1h) / ATR14_1h"),
    ("trend_alignment", "mean(signed EMA20>EMA50 across 5m,15m,1h)"),
    ("momentum_alignment", "mean(sign ROC5 across 5m,15m,1h)"),
    ("breakout_vote", "mean(breakout_up20 - breakout_down20 across timeframes)"),
    ("rsi_centered_5m", "(RSI14_5m - 50) / 50"),
    ("rsi_centered_15m", "(RSI14_15m - 50) / 50"),
    ("rsi_centered_1h", "(RSI14_1h - 50) / 50"),
    ("rsi_5m_minus_1h", "(RSI14_5m - RSI14_1h) / 100"),
    ("atr_pct_5m_to_15m", "ATRpct_5m / ATRpct_15m, clipped [0,10]"),
    ("atr_pct_15m_to_1h", "ATRpct_15m / ATRpct_1h, clipped [0,10]"),
    ("volatility_5m_to_15m", "rollingVol20_5m / rollingVol20_15m, clipped [0,10]"),
    ("range_expansion_5m_vs_15m", "rangeATR_5m / rangeATR_15m, clipped [0,10]"),
    ("ema_distance_concordance", "mean(sign distanceEMA20ATR across timeframes)"),
    ("wick_imbalance_5m", "lowerWickPct_5m - upperWickPct_5m"),
    ("body_momentum_concordance_5m", "mean(sign bodyPct, return1, ROC5)"),
    ("session_progress", "minuteOfSession / (minuteOfSession + minutesToClose)"),
    ("session_progress_sin", "sin(pi * sessionProgress)"),
    ("session_progress_cos", "cos(pi * sessionProgress)"),
    ("trend_momentum_interaction", "trendAlignment * momentumAlignment"),
    ("volatility_trend_interaction", "ATRpct_5m * trendStrength_5m"),
)
ENHANCED_FEATURE_SET_HASH = hashlib.sha256(
    json.dumps(ENHANCED_FEATURE_DEFINITIONS, separators=(",", ":")).encode("utf-8")
).hexdigest()


def enhance_dataset(dataset: DatasetBuildReport) -> DatasetBuildReport:
    samples = tuple(enhance_sample(item) for item in dataset.samples)
    names = samples[0].feature_names if samples else ()
    return replace(dataset, feature_names=names, samples=samples)


def enhance_sample(sample: TrainingSample) -> TrainingSample:
    values = {name: float(value) for name, value in zip(sample.feature_names, sample.feature_values)}
    enhanced = enhanced_values(values)
    names = sample.feature_names + tuple(name for name, _ in enhanced)
    numbers = sample.feature_values + tuple(value for _, value in enhanced)
    checksum = hashlib.sha256(
        f"{sample.input_revision_checksum}:{ENHANCED_FEATURE_SET_HASH}".encode("utf-8")
    ).hexdigest()
    return replace(
        sample,
        input_revision_checksum=checksum,
        feature_names=names,
        feature_values=numbers,
    )


def enhanced_values(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    prefixes = ("primary_5m", "context_15m", "context_1h")
    trend_strengths = tuple(
        _safe_ratio(
            values[f"{prefix}__ema_20"] - values[f"{prefix}__ema_50"],
            values[f"{prefix}__atr_14"],
        )
        for prefix in prefixes
    )
    trend_alignment = _mean(
        _signed_boolean(values[f"{prefix}__trend_ema20_above_ema50"])
        for prefix in prefixes
    )
    momentum_alignment = _mean(
        _sign(values[f"{prefix}__roc_5"]) for prefix in prefixes
    )
    breakout_vote = _mean(
        values[f"{prefix}__breakout_up_20"]
        - values[f"{prefix}__breakout_down_20"]
        for prefix in prefixes
    )
    rsi = tuple((values[f"{prefix}__rsi_14"] - 50.0) / 50.0 for prefix in prefixes)
    distance_concordance = _mean(
        _sign(values[f"{prefix}__distance_ema20_atr"]) for prefix in prefixes
    )
    body_concordance = _mean(
        _sign(values[name])
        for name in (
            "primary_5m__body_pct",
            "primary_5m__return_1",
            "primary_5m__roc_5",
        )
    )
    minute = max(0.0, values["primary_5m__minute_of_session"])
    remaining = max(0.0, values["primary_5m__minutes_to_session_close"])
    progress = minute / (minute + remaining) if minute + remaining > 0 else 0.0
    return (
        ("enhanced__trend_strength_5m", trend_strengths[0]),
        ("enhanced__trend_strength_15m", trend_strengths[1]),
        ("enhanced__trend_strength_1h", trend_strengths[2]),
        ("enhanced__trend_alignment", trend_alignment),
        ("enhanced__momentum_alignment", momentum_alignment),
        ("enhanced__breakout_vote", breakout_vote),
        ("enhanced__rsi_centered_5m", rsi[0]),
        ("enhanced__rsi_centered_15m", rsi[1]),
        ("enhanced__rsi_centered_1h", rsi[2]),
        ("enhanced__rsi_5m_minus_1h", (rsi[0] - rsi[2]) / 2.0),
        (
            "enhanced__atr_pct_5m_to_15m",
            _clipped_ratio(values["primary_5m__atr_pct"], values["context_15m__atr_pct"]),
        ),
        (
            "enhanced__atr_pct_15m_to_1h",
            _clipped_ratio(values["context_15m__atr_pct"], values["context_1h__atr_pct"]),
        ),
        (
            "enhanced__volatility_5m_to_15m",
            _clipped_ratio(
                values["primary_5m__rolling_vol_20"],
                values["context_15m__rolling_vol_20"],
            ),
        ),
        (
            "enhanced__range_expansion_5m_vs_15m",
            _clipped_ratio(values["primary_5m__range_atr"], values["context_15m__range_atr"]),
        ),
        ("enhanced__ema_distance_concordance", distance_concordance),
        (
            "enhanced__wick_imbalance_5m",
            values["primary_5m__lower_wick_pct"]
            - values["primary_5m__upper_wick_pct"],
        ),
        ("enhanced__body_momentum_concordance_5m", body_concordance),
        ("enhanced__session_progress", progress),
        ("enhanced__session_progress_sin", math.sin(math.pi * progress)),
        ("enhanced__session_progress_cos", math.cos(math.pi * progress)),
        ("enhanced__trend_momentum_interaction", trend_alignment * momentum_alignment),
        (
            "enhanced__volatility_trend_interaction",
            values["primary_5m__atr_pct"] * trend_strengths[0],
        ),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def _clipped_ratio(numerator: float, denominator: float) -> float:
    return max(0.0, min(10.0, _safe_ratio(numerator, denominator)))


def _sign(value: float) -> float:
    return 1.0 if value > 0 else -1.0 if value < 0 else 0.0


def _signed_boolean(value: float) -> float:
    return 1.0 if value >= 0.5 else -1.0


def _mean(values) -> float:
    items = tuple(values)
    return sum(items) / len(items)

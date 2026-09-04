"""Causal market-structure features layered on the locked v3 research matrix."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from nifty_terminal.domain.candle import Candle
from nifty_terminal.features.research_v3 import (
    ResearchFeatureMatrix,
    build_research_feature_matrix,
)
from nifty_terminal.ml.models import DatasetBuildReport


PRICE_ACTION_RESEARCH_VERSION = "causal_price_action_features.v1"
PRICE_ACTION_FEATURE_NAMES = (
    "price_action__confirmed_swing_high_distance_atr",
    "price_action__confirmed_swing_low_distance_atr",
    "price_action__swing_high_progression_atr",
    "price_action__swing_low_progression_atr",
    "price_action__structure_score",
    "price_action__break_of_structure_up",
    "price_action__break_of_structure_down",
    "price_action__swing_low_liquidity_sweep",
    "price_action__swing_high_liquidity_sweep",
    "price_action__range_compression_3_to_12",
    "price_action__close_location",
    "price_action__body_efficiency",
)
PRICE_ACTION_FEATURE_DEFINITION = {
    "version": PRICE_ACTION_RESEARCH_VERSION,
    "base": "stationary_price_features.v3",
    "features": PRICE_ACTION_FEATURE_NAMES,
    "pivot_confirmation": "two finalized candles on each side",
    "lookahead": "a pivot at t is first available at t+2; no developing or future candle",
    "normalization": "decision-time ATR14",
}
PRICE_ACTION_FEATURE_SET_HASH = hashlib.sha256(
    json.dumps(PRICE_ACTION_FEATURE_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class _Pivot:
    index: int
    price: float


def build_price_action_research_matrix(
    dataset: DatasetBuildReport,
    primary_candles: tuple[Candle, ...],
) -> ResearchFeatureMatrix:
    """Append price-action structure without changing or reordering v3 inputs."""
    base = build_research_feature_matrix(dataset, primary_candles)
    candle_index = {item.candle_id: index for index, item in enumerate(primary_candles)}
    base_rows = dict(zip(base.sample_ids, base.rows))
    rows = []
    for sample in dataset.samples:
        index = candle_index.get(sample.primary_candle_id)
        if index is None:
            raise ValueError(f"Primary candle missing for sample: {sample.sample_id}")
        if primary_candles[index].closes_at != sample.decision_time:
            raise ValueError("Price-action feature candle does not match sample decision time")
        base_row = base_rows[sample.sample_id]
        atr = max(_atr(primary_candles, index, 14), 1e-12)
        added = _values(primary_candles, index, atr)
        rows.append(base_row + added)
    return ResearchFeatureMatrix(
        feature_names=base.feature_names + PRICE_ACTION_FEATURE_NAMES,
        rows=tuple(rows),
        sample_ids=base.sample_ids,
    )


def _values(candles: tuple[Candle, ...], index: int, atr: float) -> tuple[float, ...]:
    # The slice ends at the decision candle.  Pivots require two right-side
    # candles, so the newest eligible pivot index is decision_index - 2.
    start = max(0, index - 80)
    highs, lows = _confirmed_pivots(candles, start, index)
    close = float(candles[index].close)
    current = candles[index]
    high_distance = (close - highs[-1].price) / atr if highs else 0.0
    low_distance = (close - lows[-1].price) / atr if lows else 0.0
    high_progression = (highs[-1].price - highs[-2].price) / atr if len(highs) >= 2 else 0.0
    low_progression = (lows[-1].price - lows[-2].price) / atr if len(lows) >= 2 else 0.0
    structure_score = (
        1.0
        if high_progression > 0 and low_progression > 0
        else -1.0
        if high_progression < 0 and low_progression < 0
        else 0.0
    )
    buffer = 0.05 * atr
    break_up = bool(highs and close > highs[-1].price + buffer)
    break_down = bool(lows and close < lows[-1].price - buffer)
    sweep_low = bool(lows and float(current.low) < lows[-1].price < close)
    sweep_high = bool(highs and float(current.high) > highs[-1].price > close)
    recent3 = candles[max(0, index - 2) : index + 1]
    recent12 = candles[max(0, index - 11) : index + 1]
    range3 = sum(float(item.high - item.low) for item in recent3) / max(len(recent3), 1)
    range12 = sum(float(item.high - item.low) for item in recent12) / max(len(recent12), 1)
    compression = range3 / max(range12, 1e-12)
    total_range = max(float(current.high - current.low), 1e-12)
    close_location = (close - float(current.low)) / total_range
    body_efficiency = float(current.close - current.open) / total_range
    values = (
        high_distance,
        low_distance,
        high_progression,
        low_progression,
        structure_score,
        float(break_up),
        float(break_down),
        float(sweep_low),
        float(sweep_high),
        compression,
        close_location,
        body_efficiency,
    )
    return tuple(_finite(item) for item in values)


def _confirmed_pivots(
    candles: tuple[Candle, ...],
    start: int,
    decision_index: int,
) -> tuple[tuple[_Pivot, ...], tuple[_Pivot, ...]]:
    highs: list[_Pivot] = []
    lows: list[_Pivot] = []
    left_bound = max(2, start)
    for index in range(left_bound, decision_index - 1):
        candle = candles[index]
        neighbors = (*candles[index - 2 : index], *candles[index + 1 : index + 3])
        if len(neighbors) != 4:
            continue
        if all(candle.high > item.high for item in neighbors):
            highs.append(_Pivot(index, float(candle.high)))
        if all(candle.low < item.low for item in neighbors):
            lows.append(_Pivot(index, float(candle.low)))
    return tuple(highs), tuple(lows)


def _atr(candles: tuple[Candle, ...], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    ranges = []
    for cursor in range(start, index + 1):
        candle = candles[cursor]
        previous_close = float(candles[cursor - 1].close) if cursor else float(candle.close)
        ranges.append(
            max(
                float(candle.high - candle.low),
                abs(float(candle.high) - previous_close),
                abs(float(candle.low) - previous_close),
            )
        )
    return sum(ranges) / max(len(ranges), 1)


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0

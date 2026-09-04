"""Stationary, causal research features for the Step 18B directional models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math

from nifty_terminal.calendar.nse import IST
from nifty_terminal.domain.candle import Candle, CandleStatus, Timeframe
from nifty_terminal.features.enhanced import enhanced_values
from nifty_terminal.ml.models import DatasetBuildReport, TrainingSample


RESEARCH_FEATURE_VERSION = "stationary_price_features.v3"
NON_STATIONARY_SUFFIXES = (
    "__sma_20",
    "__sma_50",
    "__ema_20",
    "__ema_50",
    "__atr_14",
)
ADDED_FEATURE_NAMES = (
    "research_v3__macd_line_atr",
    "research_v3__macd_signal_atr",
    "research_v3__macd_histogram_atr",
    "research_v3__adx_14",
    "research_v3__di_spread",
    "research_v3__slope_10_atr",
    "research_v3__slope_20_atr",
    "research_v3__support_distance_20_atr",
    "research_v3__resistance_distance_20_atr",
    "research_v3__support_distance_50_atr",
    "research_v3__resistance_distance_50_atr",
    "research_v3__session_return_atr",
    "research_v3__opening_gap_atr",
    "research_v3__opening_range_ready",
    "research_v3__opening_range_position",
    "research_v3__previous_high_distance_atr",
    "research_v3__previous_low_distance_atr",
    "research_v3__previous_close_distance_atr",
    "research_v3__doji",
    "research_v3__hammer",
    "research_v3__shooting_star",
    "research_v3__bullish_engulfing",
    "research_v3__bearish_engulfing",
    "research_v3__inside_bar",
    "research_v3__outside_bar",
    "research_v3__three_bar_return_atr",
    "research_v3__consecutive_direction",
)
RESEARCH_FEATURE_DEFINITION = {
    "version": RESEARCH_FEATURE_VERSION,
    "removed_absolute_price_features": list(NON_STATIONARY_SUFFIXES),
    "added_features": list(ADDED_FEATURE_NAMES),
    "inputs": "finalized 5m/15m/1h canonical candles and base feature snapshot",
    "causality": "each row uses only candles finalized by its decision time",
    "volume": "unavailable for NIFTY spot and never imputed",
}
RESEARCH_FEATURE_SET_HASH = hashlib.sha256(
    json.dumps(
        RESEARCH_FEATURE_DEFINITION, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchFeatureMatrix:
    feature_names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    sample_ids: tuple[str, ...]


def build_research_feature_matrix(
    dataset: DatasetBuildReport,
    primary_candles: tuple[Candle, ...],
) -> ResearchFeatureMatrix:
    """Build point-in-time rows aligned exactly to the dataset sample order."""
    _validate_primary_candles(primary_candles)
    candle_index = {item.candle_id: index for index, item in enumerate(primary_candles)}
    technical = _technical_series(primary_candles)
    daily = _daily_state(primary_candles)
    names: tuple[str, ...] = ()
    rows = []
    for sample in dataset.samples:
        index = candle_index.get(sample.primary_candle_id)
        if index is None:
            raise ValueError(f"Primary candle missing for sample: {sample.sample_id}")
        if primary_candles[index].closes_at != sample.decision_time:
            raise ValueError("Research feature candle does not match sample decision time")
        base = {
            name: float(value)
            for name, value in zip(sample.feature_names, sample.feature_values)
        }
        stationary = tuple(
            (name, float(value))
            for name, value in zip(sample.feature_names, sample.feature_values)
            if not name.endswith(NON_STATIONARY_SUFFIXES)
        )
        interactions = enhanced_values(base)
        added = _added_values(
            primary_candles=primary_candles,
            index=index,
            base=base,
            technical=technical,
            daily=daily,
        )
        complete = stationary + interactions + added
        current_names = tuple(name for name, _ in complete)
        if names and names != current_names:
            raise AssertionError("Research feature ordering changed inside one dataset")
        names = current_names
        values = tuple(_finite(value) for _, value in complete)
        rows.append(values)
    return ResearchFeatureMatrix(
        feature_names=names,
        rows=tuple(rows),
        sample_ids=tuple(item.sample_id for item in dataset.samples),
    )


def _added_values(
    *,
    primary_candles: tuple[Candle, ...],
    index: int,
    base: dict[str, float],
    technical: dict[str, tuple[float, ...]],
    daily: dict[int, dict[str, float]],
) -> tuple[tuple[str, float], ...]:
    candle = primary_candles[index]
    previous = primary_candles[index - 1] if index else None
    atr = max(base["primary_5m__atr_14"], 1e-12)
    state = daily[index]
    past20 = primary_candles[max(0, index - 20) : index]
    past50 = primary_candles[max(0, index - 50) : index]
    support20 = min((float(item.low) for item in past20), default=float(candle.low))
    resistance20 = max((float(item.high) for item in past20), default=float(candle.high))
    support50 = min((float(item.low) for item in past50), default=float(candle.low))
    resistance50 = max((float(item.high) for item in past50), default=float(candle.high))
    close = float(candle.close)
    body = float(candle.close - candle.open)
    body_abs = abs(body)
    total_range = max(float(candle.high - candle.low), 1e-12)
    upper = float(candle.high - max(candle.open, candle.close))
    lower = float(min(candle.open, candle.close) - candle.low)
    open_range = max(state["opening_range_high"] - state["opening_range_low"], 1e-12)
    opening_position = (
        (close - state["opening_range_low"]) / open_range
        if state["opening_range_ready"]
        else 0.5
    )
    previous_body = (
        float(previous.close - previous.open) if previous is not None else 0.0
    )
    bullish_engulfing = bool(
        previous is not None
        and previous_body < 0
        and body > 0
        and candle.open <= previous.close
        and candle.close >= previous.open
    )
    bearish_engulfing = bool(
        previous is not None
        and previous_body > 0
        and body < 0
        and candle.open >= previous.close
        and candle.close <= previous.open
    )
    directions = []
    for item in reversed(primary_candles[max(0, index - 5) : index + 1]):
        direction = 1 if item.close > item.open else -1 if item.close < item.open else 0
        if directions and direction != directions[0]:
            break
        directions.append(direction)
    consecutive = directions[0] * len(directions) if directions else 0
    return (
        ("research_v3__macd_line_atr", technical["macd_line"][index] / atr),
        ("research_v3__macd_signal_atr", technical["macd_signal"][index] / atr),
        ("research_v3__macd_histogram_atr", technical["macd_histogram"][index] / atr),
        ("research_v3__adx_14", technical["adx"][index] / 100.0),
        ("research_v3__di_spread", technical["di_spread"][index] / 100.0),
        ("research_v3__slope_10_atr", _slope(primary_candles, index, 10) / atr),
        ("research_v3__slope_20_atr", _slope(primary_candles, index, 20) / atr),
        ("research_v3__support_distance_20_atr", (close - support20) / atr),
        ("research_v3__resistance_distance_20_atr", (resistance20 - close) / atr),
        ("research_v3__support_distance_50_atr", (close - support50) / atr),
        ("research_v3__resistance_distance_50_atr", (resistance50 - close) / atr),
        ("research_v3__session_return_atr", (close - state["session_open"]) / atr),
        ("research_v3__opening_gap_atr", state["opening_gap"] / atr),
        ("research_v3__opening_range_ready", state["opening_range_ready"]),
        ("research_v3__opening_range_position", _clip(opening_position, -2.0, 3.0)),
        ("research_v3__previous_high_distance_atr", (state["previous_high"] - close) / atr),
        ("research_v3__previous_low_distance_atr", (close - state["previous_low"]) / atr),
        ("research_v3__previous_close_distance_atr", (close - state["previous_close"]) / atr),
        ("research_v3__doji", body_abs / total_range <= 0.10),
        (
            "research_v3__hammer",
            lower >= 2.0 * max(body_abs, total_range * 0.05) and upper <= body_abs,
        ),
        (
            "research_v3__shooting_star",
            upper >= 2.0 * max(body_abs, total_range * 0.05) and lower <= body_abs,
        ),
        ("research_v3__bullish_engulfing", bullish_engulfing),
        ("research_v3__bearish_engulfing", bearish_engulfing),
        (
            "research_v3__inside_bar",
            bool(previous and candle.high <= previous.high and candle.low >= previous.low),
        ),
        (
            "research_v3__outside_bar",
            bool(previous and candle.high >= previous.high and candle.low <= previous.low),
        ),
        (
            "research_v3__three_bar_return_atr",
            (
                (close - float(primary_candles[index - 3].close)) / atr
                if index >= 3
                else 0.0
            ),
        ),
        ("research_v3__consecutive_direction", float(consecutive)),
    )


def _technical_series(candles: tuple[Candle, ...]) -> dict[str, tuple[float, ...]]:
    closes = [float(item.close) for item in candles]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [
        (fast - slow) if math.isfinite(fast) and math.isfinite(slow) else 0.0
        for fast, slow in zip(ema12, ema26)
    ]
    signal = _ema(macd, 9, seed_from_start=True)
    histogram = [line - average for line, average in zip(macd, signal)]
    adx, spread = _adx(candles, 14)
    return {
        "macd_line": tuple(macd),
        "macd_signal": tuple(signal),
        "macd_histogram": tuple(histogram),
        "adx": tuple(adx),
        "di_spread": tuple(spread),
    }


def _daily_state(candles: tuple[Candle, ...]) -> dict[int, dict[str, float]]:
    by_day: dict[date, list[int]] = defaultdict(list)
    for index, candle in enumerate(candles):
        by_day[candle.opens_at.astimezone(IST).date()].append(index)
    ordered_days = sorted(by_day)
    result: dict[int, dict[str, float]] = {}
    previous_high = previous_low = previous_close = 0.0
    for day_index, day in enumerate(ordered_days):
        indices = by_day[day]
        first = candles[indices[0]]
        if day_index == 0:
            previous_high = float(first.open)
            previous_low = float(first.open)
            previous_close = float(first.open)
        opening_indices = indices[:6]
        for position, index in enumerate(indices):
            available_opening = opening_indices[: min(position + 1, 6)]
            opening_high = max(float(candles[item].high) for item in available_opening)
            opening_low = min(float(candles[item].low) for item in available_opening)
            result[index] = {
                "session_open": float(first.open),
                "opening_gap": float(first.open) - previous_close,
                "opening_range_ready": 1.0 if position >= 5 else 0.0,
                "opening_range_high": opening_high,
                "opening_range_low": opening_low,
                "previous_high": previous_high,
                "previous_low": previous_low,
                "previous_close": previous_close,
            }
        previous_high = max(float(candles[item].high) for item in indices)
        previous_low = min(float(candles[item].low) for item in indices)
        previous_close = float(candles[indices[-1]].close)
    return result


def _adx(candles: tuple[Candle, ...], period: int) -> tuple[list[float], list[float]]:
    tr = [0.0]
    plus_dm = [0.0]
    minus_dm = [0.0]
    for index in range(1, len(candles)):
        current, previous = candles[index], candles[index - 1]
        high_move = float(current.high - previous.high)
        low_move = float(previous.low - current.low)
        plus_dm.append(high_move if high_move > low_move and high_move > 0 else 0.0)
        minus_dm.append(low_move if low_move > high_move and low_move > 0 else 0.0)
        tr.append(
            max(
                float(current.high - current.low),
                abs(float(current.high - previous.close)),
                abs(float(current.low - previous.close)),
            )
        )
    smooth_tr = _wilder(tr, period)
    smooth_plus = _wilder(plus_dm, period)
    smooth_minus = _wilder(minus_dm, period)
    dx = []
    spreads = []
    for total, plus, minus in zip(smooth_tr, smooth_plus, smooth_minus):
        if total <= 1e-12:
            dx.append(0.0)
            spreads.append(0.0)
            continue
        plus_di = 100.0 * plus / total
        minus_di = 100.0 * minus / total
        spreads.append(plus_di - minus_di)
        denominator = plus_di + minus_di
        dx.append(100.0 * abs(plus_di - minus_di) / denominator if denominator else 0.0)
    return _wilder(dx, period), spreads


def _ema(values: list[float], period: int, *, seed_from_start: bool = False) -> list[float]:
    if not values:
        return []
    result = [float("nan")] * len(values)
    seed_index = 0 if seed_from_start else period - 1
    if seed_index >= len(values):
        return [0.0] * len(values)
    seed = values[0] if seed_from_start else sum(values[:period]) / period
    result[seed_index] = seed
    alpha = 2.0 / (period + 1)
    previous = seed
    for index in range(seed_index + 1, len(values)):
        previous = alpha * values[index] + (1.0 - alpha) * previous
        result[index] = previous
    return [item if math.isfinite(item) else 0.0 for item in result]


def _wilder(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    result = [0.0] * len(values)
    if len(values) < period:
        return result
    previous = sum(values[:period]) / period
    result[period - 1] = previous
    for index in range(period, len(values)):
        previous = (previous * (period - 1) + values[index]) / period
        result[index] = previous
    return result


def _slope(candles: tuple[Candle, ...], index: int, period: int) -> float:
    if index + 1 < period:
        return 0.0
    values = [float(item.close) for item in candles[index - period + 1 : index + 1]]
    x_mean = (period - 1) / 2
    y_mean = sum(values) / period
    numerator = sum((x - x_mean) * (value - y_mean) for x, value in enumerate(values))
    denominator = sum((x - x_mean) ** 2 for x in range(period))
    return numerator / denominator if denominator else 0.0


def _validate_primary_candles(candles: tuple[Candle, ...]) -> None:
    if not candles:
        raise ValueError("Research feature construction requires primary candles")
    if any(item.status is not CandleStatus.FINALIZED for item in candles):
        raise ValueError("Developing candles cannot enter research features")
    if any(item.timeframe is not Timeframe.M5 for item in candles):
        raise ValueError("Research-v3 primary features require 5m candles")
    if tuple(item.opens_at for item in candles) != tuple(
        sorted(item.opens_at for item in candles)
    ):
        raise ValueError("Research feature candles must be chronological")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _finite(value: float | bool) -> float:
    number = 1.0 if value is True else 0.0 if value is False else float(value)
    if not math.isfinite(number):
        raise ValueError("Research features must be finite")
    return number

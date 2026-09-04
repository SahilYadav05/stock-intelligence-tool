"""Causal cross-market features aligned to finalized NIFTY decisions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import json
import math

from nifty_terminal.context.bundle import ContextBundle, ContextInstrument
from nifty_terminal.domain.candle import Candle
from nifty_terminal.features.research_v3 import (
    ResearchFeatureMatrix,
    build_research_feature_matrix,
)
from nifty_terminal.ml.models import DatasetBuildReport


CONTEXT_FEATURE_VERSION = "canonical_cross_market_features.v1"
INSTRUMENT_ORDER = ("BANKNIFTY_SPOT", "INDIA_VIX_SPOT")
PER_INSTRUMENT_FEATURES = (
    "return_1",
    "return_3",
    "return_12",
    "range_pct",
    "close_location",
    "atr_pct_14",
    "realized_vol_12",
    "realized_vol_48",
    "ema_8_21_atr",
    "ema_20_50_atr",
    "rsi_14",
    "return_z_48",
)
CROSS_FEATURES = (
    "cross__bank_minus_nifty_return_1",
    "cross__bank_minus_nifty_return_3",
    "cross__bank_nifty_trend_agreement",
    "cross__vix_return_times_nifty_return",
    "cross__vix_shock_abs",
    "cross__risk_off_score",
)
CONTEXT_FEATURE_DEFINITION = {
    "version": CONTEXT_FEATURE_VERSION,
    "required_instruments": INSTRUMENT_ORDER,
    "per_instrument_features": PER_INSTRUMENT_FEATURES,
    "cross_features": CROSS_FEATURES,
    "alignment": "exact finalized 5m close; five complete source minutes required",
    "lookahead": "no backfill, interpolation, developing candle, or future join",
    "volume": "index volume remains null and is not a feature",
}
CONTEXT_FEATURE_SET_HASH = hashlib.sha256(
    json.dumps(CONTEXT_FEATURE_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextFeatureBuild:
    dataset: DatasetBuildReport
    matrix: ResearchFeatureMatrix
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Bar5:
    closes_at: datetime
    opening: float
    high: float
    low: float
    close: float


def build_context_feature_matrix(
    *,
    dataset: DatasetBuildReport,
    primary_candles: tuple[Candle, ...],
    bundle: ContextBundle,
    base_matrix: ResearchFeatureMatrix | None = None,
) -> ContextFeatureBuild:
    base = base_matrix or build_research_feature_matrix(dataset, primary_candles)
    if base.sample_ids != tuple(item.sample_id for item in dataset.samples):
        raise ValueError("Base feature matrix must align exactly to dataset samples")
    instruments = {item.instrument_id: item for item in bundle.instruments}
    missing = [name for name in INSTRUMENT_ORDER if name not in instruments]
    if missing:
        raise ValueError("Context bundle is missing required instruments: " + ", ".join(missing))
    series = {name: _aggregate_exact_5m(instruments[name]) for name in INSTRUMENT_ORDER}
    close_indices = {
        name: {row.closes_at: index for index, row in enumerate(rows)}
        for name, rows in series.items()
    }
    feature_maps = {name: _feature_series(rows) for name, rows in series.items()}
    base_by_id = dict(zip(base.sample_ids, base.rows))
    base_names = base.feature_names
    nifty_returns = _nifty_returns(primary_candles)
    nifty_by_close = {item.closes_at: index for index, item in enumerate(primary_candles)}
    retained_samples = []
    retained_rows = []
    exclusions = Counter[str]()
    context_names = tuple(
        f"context_market__{instrument.casefold()}__{feature}"
        for instrument in INSTRUMENT_ORDER
        for feature in PER_INSTRUMENT_FEATURES
    ) + CROSS_FEATURES
    for sample in dataset.samples:
        indices = {name: close_indices[name].get(sample.decision_time) for name in INSTRUMENT_ORDER}
        if any(index is None for index in indices.values()):
            exclusions["CONTEXT_5M_EXACT_CLOSE_MISSING"] += 1
            continue
        if any(int(index) < 50 for index in indices.values()):
            exclusions["CONTEXT_WARMUP_INCOMPLETE"] += 1
            continue
        nifty_index = nifty_by_close.get(sample.decision_time)
        if nifty_index is None or nifty_index < 3:
            exclusions["NIFTY_ALIGNMENT_MISSING"] += 1
            continue
        bank = feature_maps["BANKNIFTY_SPOT"][int(indices["BANKNIFTY_SPOT"])]
        vix = feature_maps["INDIA_VIX_SPOT"][int(indices["INDIA_VIX_SPOT"])]
        n1, n3, ntrend = nifty_returns[nifty_index]
        context_values = tuple(
            value
            for name in INSTRUMENT_ORDER
            for value in feature_maps[name][int(indices[name])]
        )
        bank_trend = math.copysign(1.0, bank[8]) if abs(bank[8]) > 1e-12 else 0.0
        cross = (
            bank[0] - n1,
            bank[1] - n3,
            bank_trend * ntrend,
            vix[0] * n1,
            abs(vix[11]),
            -vix[11] - ntrend,
        )
        values = tuple(_finite(value) for value in base_by_id[sample.sample_id] + context_values + cross)
        retained_samples.append(sample)
        retained_rows.append(values)
    support = Counter(item.outcome.value for item in retained_samples)
    combined_exclusions = Counter(dict(dataset.exclusion_counts))
    combined_exclusions.update(exclusions)
    filtered = replace(
        dataset,
        eligible_samples=len(retained_samples),
        outcome_support=tuple(sorted(support.items())),
        exclusion_counts=tuple(sorted(combined_exclusions.items())),
        samples=tuple(retained_samples),
    )
    names = base_names + context_names
    matrix = ResearchFeatureMatrix(
        feature_names=names,
        rows=tuple(retained_rows),
        sample_ids=tuple(item.sample_id for item in retained_samples),
    )
    coverage = {
        name: {
            "source_minutes": len(instruments[name].bars),
            "source_coverage_ratio": instruments[name].coverage_ratio,
            "complete_5m_candles": len(series[name]),
        }
        for name in INSTRUMENT_ORDER
    }
    return ContextFeatureBuild(
        dataset=filtered,
        matrix=matrix,
        diagnostics={
            "feature_version": CONTEXT_FEATURE_VERSION,
            "feature_set_hash": CONTEXT_FEATURE_SET_HASH,
            "base_feature_count": len(base_names),
            "context_feature_count": len(context_names),
            "total_feature_count": len(names),
            "input_samples": len(dataset.samples),
            "retained_samples": len(retained_samples),
            "excluded_samples": dict(exclusions),
            "instrument_coverage": coverage,
            "exact_point_in_time_join": True,
            "missing_values_imputed": False,
        },
    )


def _aggregate_exact_5m(instrument: ContextInstrument) -> tuple[_Bar5, ...]:
    by_open = {item.opens_at: item for item in instrument.bars}
    groups = []
    for last_open in sorted(by_open):
        closes_at = last_open + timedelta(minutes=1)
        if closes_at.minute % 5:
            continue
        opens = tuple(closes_at - timedelta(minutes=offset) for offset in range(5, 0, -1))
        rows = tuple(by_open.get(item) for item in opens)
        if any(row is None for row in rows):
            continue
        complete = tuple(row for row in rows if row is not None)
        groups.append(_Bar5(
            closes_at=closes_at,
            opening=float(complete[0].open),
            high=max(float(item.high) for item in complete),
            low=min(float(item.low) for item in complete),
            close=float(complete[-1].close),
        ))
    return tuple(groups)


def _feature_series(rows: tuple[_Bar5, ...]) -> tuple[tuple[float, ...], ...]:
    closes = [item.close for item in rows]
    returns = [_return(closes, index, 1) for index in range(len(rows))]
    ema8, ema21, ema20, ema50 = (_ema(closes, period) for period in (8, 21, 20, 50))
    atr = _atr(rows, 14)
    result = []
    for index, row in enumerate(rows):
        range_value = max(row.high - row.low, 1e-12)
        past12 = returns[max(0, index - 11):index + 1]
        past48 = returns[max(0, index - 47):index + 1]
        mean48 = sum(past48) / len(past48)
        std48 = _std(past48)
        result.append((
            returns[index],
            _return(closes, index, 3),
            _return(closes, index, 12),
            range_value / max(row.close, 1e-12),
            (row.close - row.low) / range_value,
            atr[index] / max(row.close, 1e-12),
            _std(past12),
            std48,
            (ema8[index] - ema21[index]) / max(atr[index], 1e-12),
            (ema20[index] - ema50[index]) / max(atr[index], 1e-12),
            _rsi(closes, index, 14) / 100.0,
            (returns[index] - mean48) / max(std48, 1e-12),
        ))
    return tuple(result)


def _nifty_returns(candles: tuple[Candle, ...]) -> dict[int, tuple[float, float, float]]:
    closes = [float(item.close) for item in candles]
    ema8, ema21 = _ema(closes, 8), _ema(closes, 21)
    return {
        index: (
            _return(closes, index, 1),
            _return(closes, index, 3),
            math.copysign(1.0, ema8[index] - ema21[index]) if abs(ema8[index] - ema21[index]) > 1e-12 else 0.0,
        )
        for index in range(len(candles))
    }


def _return(values: list[float], index: int, lag: int) -> float:
    if index < lag or values[index - lag] == 0:
        return 0.0
    return values[index] / values[index - lag] - 1.0


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _atr(rows: tuple[_Bar5, ...], period: int) -> list[float]:
    values = []
    previous = rows[0].close if rows else 0.0
    for row in rows:
        values.append(max(row.high - row.low, abs(row.high - previous), abs(row.low - previous)))
        previous = row.close
    result = []
    for index in range(len(values)):
        window = values[max(0, index - period + 1):index + 1]
        result.append(sum(window) / len(window))
    return result


def _rsi(values: list[float], index: int, period: int) -> float:
    changes = [values[item] - values[item - 1] for item in range(max(1, index - period + 1), index + 1)]
    gains = sum(max(item, 0.0) for item in changes)
    losses = sum(max(-item, 0.0) for item in changes)
    if gains + losses <= 1e-12:
        return 50.0
    return 100.0 * gains / (gains + losses)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((item - mean) ** 2 for item in values) / len(values))


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0

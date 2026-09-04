"""Point-in-time multi-timeframe training-dataset assembly."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Candle, Timeframe
from nifty_terminal.features.engine import PriceFeatureEngine
from nifty_terminal.features.models import FeatureValue, PriceFeatureRow
from nifty_terminal.ml.definitions import CLASS_ORDER
from nifty_terminal.ml.labels import FirstTouchLabelConfig, FirstTouchLabeler
from nifty_terminal.ml.models import DatasetBuildReport, TargetOutcome, TrainingSample


class TrainingDatasetAssembler:
    """Joins only context rows finalized at or before each 5m decision."""

    def __init__(
        self,
        calendar: NseSessionCalendar,
        label_config: FirstTouchLabelConfig | None = None,
    ) -> None:
        self._feature_engine = PriceFeatureEngine(calendar)
        self._labeler = FirstTouchLabeler(calendar, label_config)

    def assemble(
        self,
        *,
        dataset_id: str,
        minute_candles: tuple[Candle, ...],
        primary_candles: tuple[Candle, ...],
        context_15m_candles: tuple[Candle, ...],
        context_1h_candles: tuple[Candle, ...],
    ) -> DatasetBuildReport:
        primary_rows = self._feature_engine.calculate(primary_candles)
        context_15m_rows = self._feature_engine.calculate(context_15m_candles)
        context_1h_rows = self._feature_engine.calculate(context_1h_candles)
        labels = self._labeler.build(
            dataset_id=dataset_id,
            primary_candles=primary_candles,
            primary_features=primary_rows,
            minute_candles=minute_candles,
        )
        label_by_candle = {item.decision_candle_id: item for item in labels}
        candle_by_id = {item.candle_id: item for item in primary_candles}
        excluded = Counter[str]()
        support = Counter[str]()
        samples: list[TrainingSample] = []
        feature_names: tuple[str, ...] = ()
        context_15_index = 0
        context_1h_index = 0
        latest_15m: PriceFeatureRow | None = None
        latest_1h: PriceFeatureRow | None = None

        for primary in primary_rows:
            while (
                context_15_index < len(context_15m_rows)
                and context_15m_rows[context_15_index].decision_time <= primary.decision_time
            ):
                latest_15m = context_15m_rows[context_15_index]
                context_15_index += 1
            while (
                context_1h_index < len(context_1h_rows)
                and context_1h_rows[context_1h_index].decision_time <= primary.decision_time
            ):
                latest_1h = context_1h_rows[context_1h_index]
                context_1h_index += 1

            label = label_by_candle[primary.source_candle_id]
            if not label.eligible or label.outcome not in {
                TargetOutcome.UP,
                TargetOutcome.DOWN,
                TargetOutcome.NEITHER,
            }:
                excluded[label.exclusion_reason or "LABEL_NOT_TRAINING_ELIGIBLE"] += 1
                continue
            if not primary.is_ready:
                excluded["PRIMARY_5M_FEATURES_NOT_READY"] += 1
                continue
            if latest_15m is None or not latest_15m.is_ready:
                excluded["FINALIZED_15M_FEATURES_NOT_READY"] += 1
                continue
            if latest_1h is None or not latest_1h.is_ready:
                excluded["FINALIZED_1H_FEATURES_NOT_READY"] += 1
                continue

            values = _prefixed_values(
                ("primary_5m", primary),
                ("context_15m", latest_15m),
                ("context_1h", latest_1h),
            )
            if any(value is None for _, value in values):
                excluded["NULL_FEATURE_VALUE"] += 1
                continue
            names = tuple(name for name, _ in values)
            if feature_names and names != feature_names:
                raise ValueError("Feature ordering changed inside one training dataset")
            feature_names = names
            numeric_values = tuple(_number(value) for _, value in values)
            candle = candle_by_id[primary.source_candle_id]
            revision_checksum = hashlib.sha256(
                "|".join(
                    (
                        candle.candle_id,
                        latest_15m.source_candle_id,
                        latest_1h.source_candle_id,
                        primary.feature_set_hash,
                    )
                ).encode("utf-8")
            ).hexdigest()
            sample_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"training-sample:{dataset_id}:{label.label_id}:{revision_checksum}",
                )
            )
            samples.append(
                TrainingSample(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    instrument_id=candle.instrument_id,
                    decision_time=primary.decision_time,
                    label_window_end=label.window_ends_at,
                    label_id=label.label_id,
                    outcome=label.outcome,
                    primary_candle_id=candle.candle_id,
                    context_15m_candle_id=latest_15m.source_candle_id,
                    context_1h_candle_id=latest_1h.source_candle_id,
                    input_revision_checksum=revision_checksum,
                    feature_names=feature_names,
                    feature_values=numeric_values,
                )
            )
            support[label.outcome.value] += 1

        ambiguous = sum(item.outcome is TargetOutcome.AMBIGUOUS for item in labels)
        unavailable = sum(item.outcome is None for item in labels)
        excluded_features = sum(
            count
            for reason, count in excluded.items()
            if "FEATURE" in reason or reason == "NULL_FEATURE_VALUE"
        )
        return DatasetBuildReport(
            dataset_id=dataset_id,
            candidate_decisions=len(primary_rows),
            eligible_samples=len(samples),
            outcome_support=tuple((name, support[name]) for name in CLASS_ORDER),
            ambiguous_labels=ambiguous,
            unavailable_labels=unavailable,
            excluded_feature_rows=excluded_features,
            exclusion_counts=tuple(sorted(excluded.items())),
            feature_names=feature_names,
            labels=labels,
            samples=tuple(samples),
        )


def _prefixed_values(
    *rows: tuple[str, PriceFeatureRow],
) -> tuple[tuple[str, FeatureValue], ...]:
    return tuple(
        (f"{prefix}__{name}", value)
        for prefix, row in rows
        for name, value in row.values
    )


def _number(value: FeatureValue) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (Decimal, int)):
        return float(value)
    raise ValueError("Training features must be complete numeric or boolean values")

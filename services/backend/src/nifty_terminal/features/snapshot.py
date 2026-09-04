"""Assemble one multi-timeframe feature vector from one market snapshot."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.engine import PriceFeatureEngine
from nifty_terminal.features.models import FeatureSnapshot, FeatureValue, PriceFeatureRow


class SnapshotFeatureAssembler:
    def __init__(self, calendar: NseSessionCalendar) -> None:
        self._engine = PriceFeatureEngine(calendar)

    def assemble(self, view: MarketStateView) -> FeatureSnapshot:
        input_ids = set(view.snapshot.model_input_candle_ids)
        input_candles = tuple(
            item for item in view.finalized_candles if item.candle_id in input_ids
        )
        rows_by_timeframe = {
            timeframe: self._engine.calculate(
                tuple(item for item in input_candles if item.timeframe is timeframe)
            )
            for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1)
        }
        primary = _row_by_id(rows_by_timeframe[Timeframe.M5], view.snapshot.primary_candle_id)
        context_15m = _row_by_id(
            rows_by_timeframe[Timeframe.M15], view.snapshot.context_15m_candle_id
        )
        context_1h = _row_by_id(
            rows_by_timeframe[Timeframe.H1], view.snapshot.context_1h_candle_id
        )

        blockers: list[str] = []
        values: list[tuple[str, FeatureValue]] = []
        for prefix, row in (
            ("primary_5m", primary),
            ("context_15m", context_15m),
            ("context_1h", context_1h),
        ):
            if row is None:
                blockers.append(f"{prefix.upper()}_FEATURES_UNAVAILABLE")
                continue
            values.extend((f"{prefix}__{name}", value) for name, value in row.values)
            blockers.extend(f"{prefix.upper()}:{item}" for item in row.blockers)

        identity = (
            f"feature-snapshot:{view.snapshot.snapshot_id}:{FEATURE_VERSION}:"
            f"{FEATURE_SET_HASH}:{view.snapshot.candle_revision_checksum}"
        )
        return FeatureSnapshot(
            schema_version=1,
            feature_snapshot_id=str(uuid5(NAMESPACE_URL, identity)),
            feature_version=FEATURE_VERSION,
            feature_set_hash=FEATURE_SET_HASH,
            market_snapshot_id=view.snapshot.snapshot_id,
            instrument_id=view.snapshot.instrument_id,
            decision_time=view.snapshot.decision_time,
            input_revision_checksum=view.snapshot.candle_revision_checksum,
            values=tuple(values),
            is_ready=not blockers,
            blockers=tuple(sorted(set(blockers))),
        )


def _row_by_id(
    rows: tuple[PriceFeatureRow, ...],
    candle_id: str | None,
) -> PriceFeatureRow | None:
    if candle_id is None:
        return None
    return next((item for item in rows if item.source_candle_id == candle_id), None)

"""Point-in-time snapshot construction with strict finalized-candle gating."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.domain.candle import Candle, CandleStatus, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.snapshots.models import DataMode, MarketStateSnapshot
from nifty_terminal.snapshots.store import InMemorySnapshotStore


class MarketStateSnapshotBuilder:
    def __init__(
        self,
        *,
        candle_store: InMemoryCandleStore,
        snapshot_store: InMemorySnapshotStore,
        primary_history_limit: int = 120,
        context_history_limit: int = 64,
    ) -> None:
        self._candles = candle_store
        self._snapshots = snapshot_store
        self._history_limit = primary_history_limit
        self._context_history_limit = context_history_limit

    def build(
        self,
        *,
        primary_candle: Candle,
        created_at: datetime,
        data_mode: DataMode,
        data_status: ConnectionState,
        developing_candle: Candle | None = None,
        data_as_of: datetime | None = None,
        additional_blockers: tuple[str, ...] = (),
        persist: bool = True,
    ) -> MarketStateSnapshot:
        if primary_candle.timeframe is not Timeframe.M5:
            raise ValueError("The MVP primary snapshot candle must be 5m")
        if primary_candle.status is not CandleStatus.FINALIZED:
            raise ValueError("Official snapshots require a finalized primary candle")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if developing_candle is not None:
            if developing_candle.status is not CandleStatus.DEVELOPING:
                raise ValueError("developing_candle must have DEVELOPING status")
            if developing_candle.instrument_id != primary_candle.instrument_id:
                raise ValueError("Developing and primary candles must share an instrument")

        decision_time = primary_candle.closes_at
        resolved_data_as_of = data_as_of or decision_time
        if resolved_data_as_of.tzinfo is None or resolved_data_as_of.utcoffset() is None:
            raise ValueError("data_as_of must be timezone-aware")
        resolved_data_as_of = max(
            decision_time.astimezone(timezone.utc),
            resolved_data_as_of.astimezone(timezone.utc),
        )
        recent = self._candles.latest_series(
            primary_candle.instrument_id,
            Timeframe.M5,
            closes_at_or_before=decision_time,
        )[-self._history_limit :]
        context_15m_history = self._candles.latest_series(
            primary_candle.instrument_id,
            Timeframe.M15,
            closes_at_or_before=decision_time,
        )[-self._context_history_limit :]
        context_1h_history = self._candles.latest_series(
            primary_candle.instrument_id,
            Timeframe.H1,
            closes_at_or_before=decision_time,
        )[-self._context_history_limit :]
        context_15m = context_15m_history[-1] if context_15m_history else None
        context_1h = context_1h_history[-1] if context_1h_history else None

        model_inputs = tuple(item.candle_id for item in recent)
        model_inputs += tuple(item.candle_id for item in context_15m_history)
        model_inputs += tuple(item.candle_id for item in context_1h_history)

        blockers: list[str] = []
        if data_mode is not DataMode.LIVE:
            blockers.append("REPLAY_MODE")
        if data_status is not ConnectionState.LIVE:
            blockers.append(f"DATA_STATUS_{data_status.value}")
        if context_15m is None:
            blockers.append("FINALIZED_15M_CONTEXT_UNAVAILABLE")
        if context_1h is None:
            blockers.append("FINALIZED_1H_CONTEXT_UNAVAILABLE")
        blockers.extend(item for item in additional_blockers if item not in blockers)

        checksum = hashlib.sha256("|".join(model_inputs).encode("utf-8")).hexdigest()
        source_watermark = (
            developing_candle.source_watermark
            if developing_candle is not None
            else primary_candle.source_watermark
        )
        identity = (
            f"snapshot:{primary_candle.instrument_id}:{decision_time.isoformat()}:"
            f"{checksum}:{data_mode.value}:{data_status.value}:"
            f"{resolved_data_as_of.isoformat()}:{source_watermark}"
        )
        snapshot = MarketStateSnapshot(
            schema_version=1,
            snapshot_id=str(uuid5(NAMESPACE_URL, identity)),
            instrument_id=primary_candle.instrument_id,
            decision_time=decision_time,
            created_at=created_at.astimezone(timezone.utc),
            data_as_of=resolved_data_as_of,
            data_mode=data_mode,
            data_status=data_status,
            primary_timeframe=Timeframe.M5,
            primary_candle_id=primary_candle.candle_id,
            context_15m_candle_id=context_15m.candle_id if context_15m else None,
            context_1h_candle_id=context_1h.candle_id if context_1h else None,
            recent_primary_candle_ids=tuple(item.candle_id for item in recent),
            developing_candle_id=developing_candle.candle_id if developing_candle else None,
            model_input_candle_ids=model_inputs,
            source_watermark=source_watermark,
            candle_revision_checksum=checksum,
            live_inference_eligible=not blockers,
            blockers=tuple(blockers),
        )
        if persist:
            self._snapshots.append(snapshot)
        return snapshot

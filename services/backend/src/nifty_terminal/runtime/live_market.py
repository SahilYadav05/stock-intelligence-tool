"""Continuous canonical market-state runtime for the private NIFTY terminal."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable

from nifty_terminal.calendar.nse import MarketPhase, NseSessionCalendar
from nifty_terminal.candles.developing import DevelopingCandleEngine
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.delivery.service import MarketStateDeliveryService
from nifty_terminal.domain.candle import Candle, FinalizedMinuteBarInput, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.ledger import InMemoryEventLedger
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer
from nifty_terminal.ingestion.pipeline import IngestionPipeline, IngestionStatus
from nifty_terminal.ingestion.validator import MarketEventValidator
from nifty_terminal.providers.angelone import build_angelone_adapter
from nifty_terminal.providers.base import FinalizedMinuteProvider, ProviderAdapter
from nifty_terminal.settings import Settings
from nifty_terminal.shadow.runtime import ShadowRuntime, build_shadow_runtime
from nifty_terminal.snapshots.builder import MarketStateSnapshotBuilder
from nifty_terminal.snapshots.models import DataMode
from nifty_terminal.snapshots.store import InMemorySnapshotStore


Clock = Callable[[], datetime]
MVP_INSTRUMENT_ID = "NIFTY50_SPOT"


@dataclass(frozen=True, slots=True)
class LiveRuntimeConfig:
    history_lookback_days: int = 14
    history_recovery_minutes: int = 15
    history_poll_seconds: int = 10
    minute_finalization_delay_seconds: int = 5
    tick_fresh_seconds: int = 3
    tick_stale_seconds: int = 15
    chart_publish_interval_milliseconds: int = 250
    chart_history_primary_limit: int = 750
    chart_history_context_limit: int = 250
    chart_history_hourly_limit: int = 120
    signal_kill_switch_active: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.history_lookback_days <= 30:
            raise ValueError("history_lookback_days must be between 2 and 30")
        if self.history_recovery_minutes < 5:
            raise ValueError("history_recovery_minutes must be at least 5")
        if self.history_poll_seconds < 5:
            raise ValueError("history_poll_seconds must be at least 5")
        if self.minute_finalization_delay_seconds < 2:
            raise ValueError("minute_finalization_delay_seconds must be at least 2")
        if self.tick_fresh_seconds >= self.tick_stale_seconds:
            raise ValueError("tick_fresh_seconds must be lower than tick_stale_seconds")
        if self.chart_publish_interval_milliseconds < 100:
            raise ValueError("chart publication interval must be at least 100ms")
        if self.chart_history_primary_limit < 120:
            raise ValueError("chart_history_primary_limit must be at least 120")
        if self.chart_history_context_limit < 64:
            raise ValueError("chart_history_context_limit must be at least 64")
        if self.chart_history_hourly_limit < 32:
            raise ValueError("chart_history_hourly_limit must be at least 32")


@dataclass(frozen=True, slots=True)
class LiveRuntimeHealth:
    running: bool
    data_status: ConnectionState
    reason: str
    observed_at: datetime
    provider: str
    provider_state: ConnectionState
    last_event_time: datetime | None
    last_finalized_minute: datetime | None
    latest_snapshot_id: str | None
    raw_events_received: int
    canonical_events_stored: int
    duplicate_events: int
    quarantined_events: int
    historical_rows_ingested: int
    historical_corrections: int
    historical_rows_rejected: int
    live_auction_observations: int
    historical_auction_observations: int
    last_auction_observation_time: datetime | None
    reconnect_attempts: int
    last_error_type: str | None

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "running": self.running,
            "data_status": self.data_status.value,
            "reason": self.reason,
            "observed_at": _time(self.observed_at),
            "provider": self.provider,
            "provider_state": self.provider_state.value,
            "last_event_time": _time(self.last_event_time),
            "last_finalized_minute": _time(self.last_finalized_minute),
            "latest_snapshot_id": self.latest_snapshot_id,
            "raw_events_received": self.raw_events_received,
            "canonical_events_stored": self.canonical_events_stored,
            "duplicate_events": self.duplicate_events,
            "quarantined_events": self.quarantined_events,
            "historical_rows_ingested": self.historical_rows_ingested,
            "historical_corrections": self.historical_corrections,
            "historical_rows_rejected": self.historical_rows_rejected,
            "live_auction_observations": self.live_auction_observations,
            "historical_auction_observations": self.historical_auction_observations,
            "last_auction_observation_time": _time(self.last_auction_observation_time),
            "reconnect_attempts": self.reconnect_attempts,
            "last_error_type": self.last_error_type,
            "credentials_exposed": False,
            "automatic_trading_enabled": False,
            "nifty_spot_volume_enabled": False,
        }


class LiveMarketRuntime:
    """Owns provider recovery, canonical candles, snapshots, and publication."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter,
        delivery: MarketStateDeliveryService,
        config: LiveRuntimeConfig | None = None,
        clock: Clock | None = None,
        shadow_runtime: ShadowRuntime | None = None,
    ) -> None:
        if not isinstance(adapter, FinalizedMinuteProvider):
            raise TypeError("Live adapter must provide authoritative finalized 1m bars")
        self._adapter = adapter
        self._minute_provider = adapter
        self._delivery = delivery
        self._config = config or LiveRuntimeConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._shadow_runtime = shadow_runtime

        self._calendar = NseSessionCalendar()
        self._registry = build_mvp_instrument_registry()
        self._ledger = InMemoryEventLedger()
        self._ingestion = IngestionPipeline(
            normalizer=MarketEventNormalizer(self._registry),
            validator=MarketEventValidator(self._registry),
            ledger=self._ledger,
        )
        self._candle_store = InMemoryCandleStore()
        self._candle_engine = CandleEngine(
            calendar=self._calendar,
            registry=self._registry,
            store=self._candle_store,
        )
        self._developing = DevelopingCandleEngine(self._calendar)
        self._snapshot_store = InMemorySnapshotStore()
        self._snapshot_builder = MarketStateSnapshotBuilder(
            candle_store=self._candle_store,
            snapshot_store=self._snapshot_store,
        )

        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._supervisor_task: asyncio.Task[None] | None = None
        self._visual_flush_task: asyncio.Task[None] | None = None
        self._publish_lock = asyncio.Lock()
        self._last_publish_loop_time = 0.0
        self._data_status = ConnectionState.DISCONNECTED
        self._reason = "LIVE_RUNTIME_NOT_STARTED"
        self._last_error_type: str | None = None
        self._last_event_time: datetime | None = None
        self._last_finalized_minute: datetime | None = None
        self._latest_snapshot_id: str | None = None
        self._official_model_signature: tuple[str, ...] | None = None
        self._minute_fingerprints: dict[datetime, tuple[str, int]] = {}
        self._seen_minute_fingerprints: dict[datetime, set[str]] = {}

        self._raw_events_received = 0
        self._canonical_events_stored = 0
        self._duplicate_events = 0
        self._quarantined_events = 0
        self._historical_rows_ingested = 0
        self._historical_corrections = 0
        self._historical_rows_rejected = 0
        self._live_auction_observations = 0
        self._historical_auction_observations = 0
        self._last_auction_observation_time: datetime | None = None
        self._reconnect_attempts = 0

    @property
    def health(self) -> LiveRuntimeHealth:
        provider_health = self._adapter.health
        return LiveRuntimeHealth(
            running=self._supervisor_task is not None and not self._supervisor_task.done(),
            data_status=self._data_status,
            reason=self._reason,
            observed_at=self._now(),
            provider=self._adapter.provider_name,
            provider_state=provider_health.connection_state,
            last_event_time=self._last_event_time,
            last_finalized_minute=self._last_finalized_minute,
            latest_snapshot_id=self._latest_snapshot_id,
            raw_events_received=self._raw_events_received,
            canonical_events_stored=self._canonical_events_stored,
            duplicate_events=self._duplicate_events,
            quarantined_events=self._quarantined_events,
            historical_rows_ingested=self._historical_rows_ingested,
            historical_corrections=self._historical_corrections,
            historical_rows_rejected=self._historical_rows_rejected,
            live_auction_observations=self._live_auction_observations,
            historical_auction_observations=self._historical_auction_observations,
            last_auction_observation_time=self._last_auction_observation_time,
            reconnect_attempts=self._reconnect_attempts,
            last_error_type=self._last_error_type,
        )

    @property
    def shadow_status(self) -> dict[str, object] | None:
        return (
            self._shadow_runtime.status.to_contract()
            if self._shadow_runtime is not None
            else None
        )

    def chart_history(self, *, decision_time: datetime) -> tuple[Candle, ...]:
        """Return chart-only history from the same canonical candle store.

        Every returned candle is finalized at or before the requested decision
        time. These extra display candles are deliberately not added to the
        versioned model-input list.
        """

        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise ValueError("decision_time must be timezone-aware")
        cutoff = decision_time.astimezone(timezone.utc)
        candles: list[Candle] = []
        for timeframe, limit in (
            (Timeframe.M5, self._config.chart_history_primary_limit),
            (Timeframe.M15, self._config.chart_history_context_limit),
            (Timeframe.H1, self._config.chart_history_hourly_limit),
        ):
            candles.extend(
                self._candle_store.latest_series(
                    MVP_INSTRUMENT_ID,
                    timeframe,
                    closes_at_or_before=cutoff,
                )[-limit:]
            )
        return tuple(candles)

    async def start(self) -> None:
        if self._supervisor_task is not None and not self._supervisor_task.done():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._data_status = ConnectionState.CONNECTING
        self._reason = "CONNECTING_TO_ANGEL_ONE"
        self._supervisor_task = asyncio.create_task(
            self._supervise(),
            name="live-market-supervisor",
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._visual_flush_task is not None:
            self._visual_flush_task.cancel()
        task = self._supervisor_task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=6)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._supervisor_task = None
        self._data_status = ConnectionState.DISCONNECTED
        self._reason = "LIVE_RUNTIME_STOPPED"

    async def wait_until_ready(self, timeout: float = 30) -> None:
        await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)

    async def _supervise(self) -> None:
        retry_delay = 1.0
        while not self._stop_event.is_set():
            children: list[asyncio.Task[object]] = []
            try:
                self._data_status = ConnectionState.CONNECTING
                self._reason = "CONNECTING_TO_ANGEL_ONE"
                await self._adapter.connect()
                changed = await self._recover_history(startup=True)
                if not changed and self._latest_primary() is None:
                    raise RuntimeError("Angel One returned no usable finalized NIFTY candles")
                self._refresh_status()
                await self._publish_current(persist=True)
                self._ready_event.set()
                self._last_error_type = None
                retry_delay = 1.0

                children = [
                    asyncio.create_task(self._consume_stream(), name="live-market-stream"),
                    asyncio.create_task(self._finalizer_loop(), name="live-minute-finalizer"),
                    asyncio.create_task(self._freshness_loop(), name="live-data-freshness"),
                    asyncio.create_task(self._stop_event.wait(), name="live-runtime-stop"),
                ]
                done, _ = await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
                if self._stop_event.is_set():
                    return
                for completed in done:
                    error = completed.exception()
                    if error is not None:
                        raise error
                raise RuntimeError("A live-runtime worker stopped unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._last_error_type = type(error).__name__
                self._data_status = ConnectionState.DISCONNECTED
                self._reason = f"LIVE_RUNTIME_RECOVERY_{type(error).__name__.upper()}"
                self._reconnect_attempts += 1
                await self._publish_current(persist=False)
            finally:
                for child in children:
                    child.cancel()
                if children:
                    await asyncio.gather(*children, return_exceptions=True)
                await self._adapter.disconnect()

            if self._stop_event.is_set():
                return
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=retry_delay)
            except TimeoutError:
                pass
            retry_delay = min(retry_delay * 2, 30.0)

    async def _recover_history(self, *, startup: bool) -> bool:
        now = self._now()
        end = now - timedelta(seconds=self._config.minute_finalization_delay_seconds)
        if startup or self._last_finalized_minute is None:
            start = now - timedelta(days=self._config.history_lookback_days)
        else:
            start = min(
                self._last_finalized_minute
                - timedelta(minutes=self._config.history_recovery_minutes),
                end - timedelta(minutes=self._config.history_recovery_minutes),
            )
        if start >= end:
            return False
        bars = await self._minute_provider.fetch_finalized_minutes(
            from_time=start,
            to_time=end,
        )
        return self._ingest_finalized_minutes(bars)

    def _ingest_finalized_minutes(
        self,
        bars: tuple[FinalizedMinuteBarInput, ...],
    ) -> bool:
        changed = False
        for incoming in sorted(bars, key=lambda item: item.opens_at):
            if self._calendar.market_phase(incoming.opens_at).is_closing_auction:
                self._historical_auction_observations += 1
                self._last_auction_observation_time = incoming.opens_at
                continue
            key = incoming.opens_at.astimezone(timezone.utc)
            previous = self._minute_fingerprints.get(key)
            if previous is not None and previous[0] == incoming.provider_bar_id:
                continue
            seen = self._seen_minute_fingerprints.setdefault(key, set())
            if incoming.provider_bar_id in seen:
                self._historical_rows_rejected += 1
                continue
            revision = 1 if previous is None else previous[1] + 1
            bar = replace(incoming, provider_revision=revision)
            try:
                self._candle_engine.ingest_finalized_minute(bar)
            except ValueError:
                self._historical_rows_rejected += 1
                continue
            seen.add(incoming.provider_bar_id)
            self._minute_fingerprints[key] = (incoming.provider_bar_id, revision)
            self._historical_rows_ingested += 1
            if previous is not None:
                self._historical_corrections += 1
            self._last_finalized_minute = max(
                self._last_finalized_minute or bar.closes_at,
                bar.closes_at,
            )
            changed = True
        return changed

    async def _consume_stream(self) -> None:
        async for raw in self._adapter.stream():
            self._raw_events_received += 1
            outcome = self._ingestion.process(raw)
            if outcome.status is IngestionStatus.DUPLICATE:
                self._duplicate_events += 1
                continue
            if outcome.status is IngestionStatus.QUARANTINED or outcome.event is None:
                self._quarantined_events += 1
                continue
            self._canonical_events_stored += 1
            event = outcome.event
            self._last_event_time = event.normalized_event_time
            if self._calendar.market_phase(event.normalized_event_time).is_closing_auction:
                self._live_auction_observations += 1
                self._last_auction_observation_time = event.normalized_event_time
                self._refresh_status()
                self._schedule_visual_publish()
                continue
            try:
                self._developing.apply(event, Timeframe.M1)
                self._developing.apply(event, Timeframe.M5)
            except ValueError:
                # Pre-open, post-close, or out-of-order ticks never mutate chart state.
                self._quarantined_events += 1
                continue
            self._refresh_status()
            self._schedule_visual_publish()
        if not self._stop_event.is_set():
            raise RuntimeError("Angel One market stream ended unexpectedly")

    async def _finalizer_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self._config.history_poll_seconds)
            changed = await self._recover_history(startup=False)
            self._refresh_status()
            if changed and self._model_signature() != self._official_model_signature:
                await self._publish_current(persist=True)

    async def _freshness_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(1)
            previous = self._data_status
            self._refresh_status()
            if self._data_status is not previous:
                await self._publish_current(persist=False)

    def _schedule_visual_publish(self) -> None:
        if self._visual_flush_task is not None and not self._visual_flush_task.done():
            return
        loop = asyncio.get_running_loop()
        interval = self._config.chart_publish_interval_milliseconds / 1_000
        delay = max(0.0, interval - (loop.time() - self._last_publish_loop_time))
        self._visual_flush_task = asyncio.create_task(
            self._flush_visual_after(delay),
            name="live-chart-publisher",
        )

    async def _flush_visual_after(self, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        await self._publish_current(persist=False)

    async def _publish_current(self, *, persist: bool) -> None:
        primary = self._latest_primary()
        if primary is None:
            return
        async with self._publish_lock:
            now = self._now()
            developing = self._developing.current(MVP_INSTRUMENT_ID, Timeframe.M5)
            if developing is not None and developing.opens_at < primary.closes_at:
                developing = None
            blockers = self._session_blockers(primary, now)
            snapshot = self._snapshot_builder.build(
                primary_candle=primary,
                created_at=now,
                data_mode=DataMode.LIVE,
                data_status=self._data_status,
                developing_candle=developing,
                data_as_of=self._last_event_time or primary.closes_at,
                additional_blockers=blockers,
                persist=persist,
            )
            required_ids = set(snapshot.model_input_candle_ids)
            finalized: list[Candle] = []
            for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
                finalized.extend(
                    item
                    for item in self._candle_store.latest_series(
                        MVP_INSTRUMENT_ID,
                        timeframe,
                        closes_at_or_before=snapshot.decision_time,
                    )
                    if item.candle_id in required_ids
                )
            view = MarketStateView(
                schema_version=1,
                snapshot=snapshot,
                finalized_candles=tuple(finalized),
                developing_candle=developing,
                published_at=now,
            )
            await self._delivery.publish(view)
            if persist and self._shadow_runtime is not None:
                minute_candles = self._candle_store.latest_series(
                    MVP_INSTRUMENT_ID,
                    Timeframe.M1,
                    closes_at_or_before=snapshot.decision_time,
                )
                await asyncio.to_thread(
                    self._shadow_runtime.process,
                    view=view,
                    minute_candles=minute_candles,
                    observed_at=now,
                )
            self._latest_snapshot_id = snapshot.snapshot_id
            if persist:
                self._official_model_signature = tuple(snapshot.model_input_candle_ids)
            self._last_publish_loop_time = asyncio.get_running_loop().time()

    def _latest_primary(self) -> Candle | None:
        series = self._candle_store.latest_series(MVP_INSTRUMENT_ID, Timeframe.M5)
        return series[-1] if series else None

    def _model_signature(self) -> tuple[str, ...]:
        primary = self._latest_primary()
        if primary is None:
            return ()
        decision_time = primary.closes_at
        signature: list[str] = []
        for timeframe, limit in (
            (Timeframe.M5, 120),
            (Timeframe.M15, 64),
            (Timeframe.H1, 64),
        ):
            signature.extend(
                item.candle_id
                for item in self._candle_store.latest_series(
                    MVP_INSTRUMENT_ID,
                    timeframe,
                    closes_at_or_before=decision_time,
                )[-limit:]
            )
        return tuple(signature)

    def _refresh_status(self) -> None:
        now = self._now()
        phase = self._calendar.market_phase(now)
        if phase.is_closing_auction:
            self._data_status = ConnectionState.MARKET_CLOSED
            self._reason = "NSE_CLOSING_AUCTION_ACTIVE_STANDARD_SIGNAL_DISABLED"
            return
        session = self._calendar.session_containing(now)
        if session is None:
            self._data_status = ConnectionState.MARKET_CLOSED
            self._reason = "NSE_REGULAR_SESSION_CLOSED"
            return
        provider_state = self._adapter.health.connection_state
        if provider_state is not ConnectionState.LIVE:
            self._data_status = provider_state
            self._reason = f"PROVIDER_{provider_state.value}"
            return
        if self._last_event_time is None:
            self._data_status = ConnectionState.RECOVERING
            self._reason = "WAITING_FOR_FIRST_LIVE_TICK"
            return
        tick_age = max(0.0, (now - self._last_event_time).total_seconds())
        if tick_age > self._config.tick_stale_seconds:
            self._data_status = ConnectionState.STALE
            self._reason = "LIVE_TICK_STALE"
            return
        if tick_age > self._config.tick_fresh_seconds:
            self._data_status = ConnectionState.DELAYED
            self._reason = "LIVE_TICK_DELAYED"
            return

        expected_close = (
            now - timedelta(seconds=self._config.minute_finalization_delay_seconds)
        ).replace(second=0, microsecond=0)
        if expected_close > session.opens_at.astimezone(timezone.utc):
            if self._last_finalized_minute is None:
                self._data_status = ConnectionState.STALE
                self._reason = "FINALIZED_MINUTE_UNAVAILABLE"
                return
            lag = (expected_close - self._last_finalized_minute).total_seconds()
            if lag >= 300:
                self._data_status = ConnectionState.STALE
                self._reason = "FINALIZED_MINUTES_STALE"
                return
            if lag >= 60:
                self._data_status = ConnectionState.DELAYED
                self._reason = "FINALIZED_MINUTE_DELAYED"
                return
        self._data_status = ConnectionState.LIVE
        self._reason = "CANONICAL_MARKET_STATE_FRESH"

    def _session_blockers(self, primary: Candle, now: datetime) -> tuple[str, ...]:
        blockers: list[str] = []
        phase = self._calendar.market_phase(now)
        if phase.is_closing_auction:
            blockers.append("CLOSING_AUCTION_STANDARD_SIGNAL_DISABLED")
        if self._config.signal_kill_switch_active:
            blockers.append("LIVE_SIGNAL_KILL_SWITCH_ACTIVE")
        session = self._calendar.session_containing(now)
        if session is None:
            return tuple(blockers)
        session_open = session.opens_at.astimezone(timezone.utc)
        if primary.closes_at <= session_open:
            blockers.append("PRIMARY_5M_CANDLE_NOT_FINALIZED_FOR_CURRENT_SESSION")
        if primary.closes_at + timedelta(minutes=60) > session.closes_at.astimezone(
            timezone.utc
        ):
            blockers.append("PREDICTION_HORIZON_CROSSES_CONTINUOUS_SESSION_CLOSE")
        return tuple(blockers)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Live runtime clock must be timezone-aware")
        return value.astimezone(timezone.utc)


def build_angelone_live_runtime(
    *,
    settings: Settings,
    delivery: MarketStateDeliveryService,
) -> LiveMarketRuntime:
    if settings.market_data_mode != "live" or settings.market_data_provider != "angelone":
        raise ValueError("Angel One live runtime requires live/angelone settings")
    if not settings.angelone_credentials_configured:
        raise ValueError("Angel One credentials are incomplete")
    adapter = build_angelone_adapter(settings)
    shadow_runtime = None
    if settings.shadow_mode_enabled:
        assert settings.shadow_runtime_manifest_path is not None
        shadow_runtime = build_shadow_runtime(
            manifest_path=settings.shadow_runtime_manifest_path,
            ledger_path=settings.shadow_ledger_path,
        )
    return LiveMarketRuntime(
        adapter=adapter,
        delivery=delivery,
        config=LiveRuntimeConfig(
            history_lookback_days=settings.live_history_lookback_days,
            history_recovery_minutes=settings.live_history_recovery_minutes,
            history_poll_seconds=settings.live_history_poll_seconds,
            minute_finalization_delay_seconds=(
                settings.live_minute_finalization_delay_seconds
            ),
            tick_fresh_seconds=settings.live_tick_fresh_seconds,
            tick_stale_seconds=settings.live_tick_stale_seconds,
            chart_publish_interval_milliseconds=(
                settings.live_chart_publish_interval_milliseconds
            ),
            chart_history_primary_limit=settings.live_chart_history_primary_limit,
            chart_history_context_limit=settings.live_chart_history_context_limit,
            chart_history_hourly_limit=settings.live_chart_history_hourly_limit,
            signal_kill_switch_active=settings.live_signal_kill_switch,
        ),
        shadow_runtime=shadow_runtime,
    )


def _time(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None

"""Historical import pipeline reusing the canonical Step 3 candle engine."""

from __future__ import annotations

from dataclasses import replace
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.domain.candle import Candle
from nifty_terminal.domain.instruments import InstrumentRegistry
from nifty_terminal.history.models import (
    HistoricalImportResult,
    HistoricalBatch,
    HistoricalQualityReport,
    HistoricalRequest,
    QualityStatus,
)
from nifty_terminal.history.quality import HistoricalQualityInspector
from nifty_terminal.history.repository import HistoricalRepository
from nifty_terminal.history.sources.base import HistoricalDataSource


class HistoricalImportPipeline:
    def __init__(
        self,
        *,
        source: HistoricalDataSource,
        repository: HistoricalRepository,
        calendar: NseSessionCalendar,
        registry: InstrumentRegistry,
    ) -> None:
        self._source = source
        self._repository = repository
        self._calendar = calendar
        self._registry = registry

    def run(self, request: HistoricalRequest) -> HistoricalImportResult:
        batch = self._source.fetch(request)
        quality = HistoricalQualityInspector(self._calendar).inspect(batch)
        dataset_id = _dataset_id(batch)
        candles: list[Candle] = []

        if quality.status is not QualityStatus.REJECTED:
            store = InMemoryCandleStore()
            engine = CandleEngine(
                calendar=self._calendar,
                registry=self._registry,
                store=store,
            )
            try:
                for row in batch.rows:
                    result = engine.ingest_finalized_minute(row)
                    candles.append(result.minute_candle)
                    candles.extend(result.finalized_candles)
            except ValueError as error:
                quality = _reject(quality, f"CANONICAL_CANDLE_REJECTION:{error}")
                candles.clear()

        imported = self._repository.save_dataset(
            dataset_id=dataset_id,
            batch=batch,
            quality=quality,
            candles=tuple(candles),
        )
        return HistoricalImportResult(
            dataset_id=dataset_id,
            imported=imported,
            candle_revision_count=len(candles) if imported else 0,
            quality=quality,
        )


def _dataset_id(batch: HistoricalBatch) -> str:
    identity = (
        f"historical-dataset:{batch.provider}:{batch.request.instrument_id}:"
        f"{batch.request.starts_at.isoformat()}:{batch.request.ends_at.isoformat()}:"
        f"{batch.source_sha256}"
    )
    return str(uuid5(NAMESPACE_URL, identity))


def _reject(
    quality: HistoricalQualityReport,
    error: str,
) -> HistoricalQualityReport:
    return replace(
        quality,
        status=QualityStatus.REJECTED,
        errors=tuple(sorted(set(quality.errors + (error,)))),
    )

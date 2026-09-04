"""End-to-end historical labeling, training, validation, and replay pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Candle, Timeframe
from nifty_terminal.history.models import QualityStatus
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.models import (
    MLResearchPipelineResult,
    TrainingRunReport,
    WalkForwardConfig,
)
from nifty_terminal.ml.training import MLResearchRunner


class MLResearchRepository(Protocol):
    def load_dataset_quality_status(self, *, dataset_id: str) -> QualityStatus | None: ...

    def load_latest_candles(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        timeframe: Timeframe,
    ) -> tuple[Candle, ...]: ...

    def save_ml_research_run(self, report: TrainingRunReport) -> bool: ...


class MLResearchPipeline:
    def __init__(
        self,
        *,
        repository: MLResearchRepository,
        calendar: NseSessionCalendar,
    ) -> None:
        self._repository = repository
        self._assembler = TrainingDatasetAssembler(calendar)
        self._runner = MLResearchRunner()

    def run(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        config: WalkForwardConfig,
        created_at: datetime | None = None,
    ) -> MLResearchPipelineResult:
        quality = self._repository.load_dataset_quality_status(dataset_id=dataset_id)
        if quality is None:
            raise ValueError("Historical dataset does not exist")
        if quality is not QualityStatus.PASS:
            raise ValueError(
                f"ML research requires a PASS dataset; received {quality.value}"
            )
        candles = {
            timeframe: self._repository.load_latest_candles(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                timeframe=timeframe,
            )
            for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1)
        }
        if any(not candles[timeframe] for timeframe in candles):
            raise ValueError("Dataset is missing one or more required canonical timeframes")
        dataset_report = self._assembler.assemble(
            dataset_id=dataset_id,
            minute_candles=candles[Timeframe.M1],
            primary_candles=candles[Timeframe.M5],
            context_15m_candles=candles[Timeframe.M15],
            context_1h_candles=candles[Timeframe.H1],
        )
        report = self._runner.run(
            dataset_report=dataset_report,
            config=config,
            created_at=created_at,
        )
        return MLResearchPipelineResult(
            report=report,
            persisted=self._repository.save_ml_research_run(report),
        )

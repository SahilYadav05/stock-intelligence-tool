from __future__ import annotations

from contextlib import closing
from datetime import timedelta
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from history_fixture import SESSION_OPEN, write_history_csv
from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.features.engine import PriceFeatureEngine
from nifty_terminal.features.materializer import HistoricalFeatureMaterializer
from nifty_terminal.history.models import HistoricalRequest, QualityStatus
from nifty_terminal.history.pipeline import HistoricalImportPipeline
from nifty_terminal.history.sources.csv_source import CsvHistoricalDataSource
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository


class HistoricalPipelineTests(TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.csv_path = root / "provider-export.csv"
        self.database_path = root / "research.sqlite3"
        self.repository = SQLiteHistoricalRepository(self.database_path)
        self.calendar = NseSessionCalendar()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(self, count: int = 15) -> HistoricalRequest:
        return HistoricalRequest(
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M1,
            starts_at=SESSION_OPEN,
            ends_at=SESSION_OPEN + timedelta(minutes=count),
        )

    def pipeline(self) -> HistoricalImportPipeline:
        return HistoricalImportPipeline(
            source=CsvHistoricalDataSource(path=self.csv_path, provider="provider-export"),
            repository=self.repository,
            calendar=self.calendar,
            registry=build_mvp_instrument_registry(),
        )

    def test_complete_csv_is_versioned_aggregated_and_idempotent(self) -> None:
        write_history_csv(self.csv_path, list(range(15)))
        first = self.pipeline().run(self.request())
        second = self.pipeline().run(self.request())

        self.assertEqual(first.quality.status, QualityStatus.PASS)
        self.assertTrue(first.imported)
        self.assertFalse(second.imported)
        self.assertEqual(first.candle_revision_count, 19)
        self.assertEqual(
            len(
                self.repository.load_latest_candles(
                    dataset_id=first.dataset_id,
                    instrument_id="NIFTY50_SPOT",
                    timeframe=Timeframe.M5,
                )
            ),
            3,
        )

    def test_missing_minute_is_stored_as_degraded_not_silently_filled(self) -> None:
        indexes = [0, 1, 2, 4, 5]
        write_history_csv(self.csv_path, indexes)
        result = self.pipeline().run(self.request(count=6))

        self.assertEqual(result.quality.status, QualityStatus.DEGRADED)
        self.assertEqual(result.quality.missing_minutes, 1)
        five_minute = self.repository.load_latest_candles(
            dataset_id=result.dataset_id,
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M5,
        )
        self.assertEqual(five_minute, ())

    def test_missing_leading_minutes_are_included_in_quality_coverage(self) -> None:
        write_history_csv(self.csv_path, [2, 3, 4])
        result = self.pipeline().run(self.request(count=5))

        self.assertEqual(result.quality.status, QualityStatus.DEGRADED)
        self.assertEqual(result.quality.missing_minutes, 2)

    def test_fake_nifty_volume_rejects_entire_dataset(self) -> None:
        write_history_csv(self.csv_path, list(range(5)), volume="100")
        result = self.pipeline().run(self.request(count=5))

        self.assertEqual(result.quality.status, QualityStatus.REJECTED)
        self.assertEqual(result.candle_revision_count, 0)
        self.assertTrue(any("Volume" in item for item in result.quality.errors))

    def test_late_provider_correction_creates_immutable_latest_revision(self) -> None:
        write_history_csv(self.csv_path, list(range(5)), correction_after=True)
        result = self.pipeline().run(self.request(count=5))
        minute = self.repository.load_latest_candles(
            dataset_id=result.dataset_id,
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M1,
        )[0]
        five = self.repository.load_latest_candles(
            dataset_id=result.dataset_id,
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M5,
        )[0]

        self.assertEqual(result.quality.correction_rows, 1)
        self.assertEqual(minute.revision, 2)
        self.assertEqual(five.revision, 2)

    def test_repository_tables_are_append_only(self) -> None:
        write_history_csv(self.csv_path, list(range(5)))
        self.pipeline().run(self.request(count=5))
        with closing(sqlite3.connect(self.database_path)) as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute("UPDATE historical_datasets SET provider = 'changed'")

    def test_repository_connection_context_closes_its_file_handle(self) -> None:
        with self.repository._connect() as connection:
            connection.execute("SELECT 1").fetchone()

        with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
            connection.execute("SELECT 1")

    def test_feature_materialization_uses_repository_candles(self) -> None:
        write_history_csv(self.csv_path, list(range(60)))
        result = self.pipeline().run(self.request(count=60))
        materializer = HistoricalFeatureMaterializer(
            repository=self.repository,
            engine=PriceFeatureEngine(self.calendar),
        )
        inserted = materializer.run(
            dataset_id=result.dataset_id,
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M5,
        )

        self.assertEqual(inserted, 12)

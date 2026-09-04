from __future__ import annotations

from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.ledger import InMemoryEventLedger
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer
from nifty_terminal.ingestion.pipeline import IngestionPipeline, IngestionStatus
from nifty_terminal.ingestion.validator import MarketEventValidator
from nifty_terminal.providers.replay import ReplayProviderAdapter
from factories import raw_index_event


def build_pipeline(ledger: InMemoryEventLedger) -> IngestionPipeline:
    registry = build_mvp_instrument_registry()
    return IngestionPipeline(
        normalizer=MarketEventNormalizer(registry),
        validator=MarketEventValidator(registry),
        ledger=ledger,
    )


class PipelineTests(TestCase):
    def test_duplicate_event_is_not_appended_twice(self) -> None:
        ledger = InMemoryEventLedger()
        pipeline = build_pipeline(ledger)

        first = pipeline.process(raw_index_event(connection_epoch="socket-a"))
        duplicate = pipeline.process(raw_index_event(connection_epoch="socket-b"))

        self.assertEqual(first.status, IngestionStatus.STORED)
        self.assertEqual(duplicate.status, IngestionStatus.DUPLICATE)
        self.assertEqual(len(ledger), 1)

    def test_unknown_provider_symbol_is_quarantined(self) -> None:
        ledger = InMemoryEventLedger()
        pipeline = build_pipeline(ledger)

        outcome = pipeline.process(raw_index_event(provider_instrument_id="UNKNOWN"))

        self.assertEqual(outcome.status, IngestionStatus.QUARANTINED)
        self.assertEqual(len(ledger), 0)

    def test_contiguous_sequence_gap_is_quarantined(self) -> None:
        ledger = InMemoryEventLedger()
        pipeline = build_pipeline(ledger)

        first = pipeline.process(raw_index_event(provider_sequence=1001))
        gap = pipeline.process(
            raw_index_event(
                provider_sequence=1003,
                raw_payload={"fixture": True, "sequence": 1003},
            )
        )

        self.assertEqual(first.status, IngestionStatus.STORED)
        self.assertEqual(gap.status, IngestionStatus.QUARANTINED)
        self.assertIn("SEQUENCE_GAP", {item.code for item in gap.issues})
        self.assertEqual(len(ledger), 1)


class ReplayPipelineTests(IsolatedAsyncioTestCase):
    async def test_jsonl_replay_runs_through_the_same_ingestion_pipeline(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "nifty50_replay.jsonl"
        adapter = ReplayProviderAdapter.from_jsonl(fixture)
        ledger = InMemoryEventLedger()
        pipeline = build_pipeline(ledger)

        outcomes = await pipeline.consume(adapter)

        self.assertEqual(
            [item.status for item in outcomes],
            [
                IngestionStatus.STORED,
                IngestionStatus.STORED,
                IngestionStatus.DUPLICATE,
            ],
        )
        self.assertEqual(len(ledger), 2)
        self.assertEqual(adapter.health.connection_state, ConnectionState.DISCONNECTED)

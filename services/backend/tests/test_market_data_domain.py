from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest import TestCase

from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer
from factories import raw_index_event


class MarketDataDomainTests(TestCase):
    def setUp(self) -> None:
        self.registry = build_mvp_instrument_registry()
        self.normalizer = MarketEventNormalizer(self.registry)

    def test_replay_symbol_resolves_to_canonical_nifty_index(self) -> None:
        instrument = self.registry.resolve("replay", "NIFTY50_TEST")

        self.assertEqual(instrument.instrument_id, "NIFTY50_SPOT")
        self.assertFalse(instrument.volume_supported)

    def test_event_identity_is_deterministic_across_reconnects(self) -> None:
        first = self.normalizer.normalize(raw_index_event(connection_epoch="socket-a"))
        repeated = self.normalizer.normalize(
            raw_index_event(connection_epoch="socket-b")
        )

        self.assertEqual(first.event_id, repeated.event_id)
        self.assertEqual(first.deduplication_key, repeated.deduplication_key)

    def test_canonical_events_are_immutable(self) -> None:
        event = self.normalizer.normalize(raw_index_event())

        with self.assertRaises(FrozenInstanceError):
            event.price = None  # type: ignore[misc]

    def test_contract_serializes_decimal_as_text(self) -> None:
        event = self.normalizer.normalize(raw_index_event())

        payload = event.to_contract()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["instrument_id"], "NIFTY50_SPOT")
        self.assertEqual(payload["price"], "25000.00")
        self.assertIsNone(payload["cumulative_volume"])

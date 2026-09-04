from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest import TestCase

from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer
from nifty_terminal.ingestion.validator import MarketEventValidator
from factories import raw_index_event


class ValidationTests(TestCase):
    def setUp(self) -> None:
        self.registry = build_mvp_instrument_registry()
        self.normalizer = MarketEventNormalizer(self.registry)
        self.validator = MarketEventValidator(self.registry)

    def test_valid_nifty_index_event_is_accepted(self) -> None:
        event = self.normalizer.normalize(raw_index_event())

        decision = self.validator.validate(event)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.issues, ())

    def test_nifty_spot_volume_is_rejected(self) -> None:
        event = self.normalizer.normalize(
            raw_index_event(cumulative_volume=Decimal("1000"))
        )

        decision = self.validator.validate(event)

        self.assertFalse(decision.accepted)
        self.assertIn("VOLUME_NOT_SUPPORTED", {item.code for item in decision.issues})

    def test_future_event_is_rejected(self) -> None:
        raw = raw_index_event()
        event = self.normalizer.normalize(
            raw_index_event(
                provider_event_time=raw.server_arrival_time + timedelta(seconds=5)
            )
        )

        decision = self.validator.validate(event)

        self.assertFalse(decision.accepted)
        self.assertIn("EVENT_FROM_FUTURE", {item.code for item in decision.issues})

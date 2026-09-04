from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.ledger import InMemoryEventLedger
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer
from nifty_terminal.ingestion.pipeline import IngestionPipeline, IngestionStatus
from nifty_terminal.ingestion.validator import MarketEventValidator
from nifty_terminal.providers.angelone import (
    AngelOneConfig,
    AngelOneCredentials,
    AngelOneProviderAdapter,
)
from nifty_terminal.settings import Settings


NOW = datetime(2026, 8, 24, 4, 0, 1, tzinfo=timezone.utc)


class FakeTotp:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def now(self) -> str:
        return "123456"


class FakeSmartConnect:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def generateSession(self, client_code: str, pin: str, totp: str):
        assert client_code == "CLIENT123"
        assert pin == "1234"
        assert totp == "123456"
        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {"jwtToken": "jwt", "refreshToken": "refresh"},
        }

    def getfeedToken(self) -> str:
        return "feed"

    def getCandleData(self, request: dict[str, object]):
        assert request["exchange"] == "NSE"
        assert request["symboltoken"] == "99926000"
        assert request["interval"] == "ONE_MINUTE"
        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": [
                [
                    "2026-08-24T09:15:00+05:30",
                    25000,
                    25005,
                    24998,
                    25003,
                    999999,
                ]
            ],
        }


class FakeWebSocket:
    last_instance = None

    def __init__(self, auth: str, api_key: str, client_code: str, feed: str) -> None:
        self.auth = auth
        self.api_key = api_key
        self.client_code = client_code
        self.feed = feed
        self.on_open = None
        self.on_data = None
        self.on_error = None
        self.on_close = None
        self.subscriptions: list[tuple[object, ...]] = []
        FakeWebSocket.last_instance = self

    def subscribe(self, correlation_id: str, mode: int, token_list: list[dict]):
        self.subscriptions.append((correlation_id, mode, token_list))

    def connect(self) -> None:
        self.on_open(self)
        self.on_data(
            self,
            {
                "subscription_mode": 1,
                "exchange_type": 1,
                "token": "99926000",
                "sequence_number": 8751,
                "exchange_timestamp": int(NOW.timestamp() * 1000),
                "last_traded_price": 2500025,
                "volume_trade_for_the_day": 999999,
            },
        )

    def close_connection(self) -> None:
        if self.on_close is not None:
            self.on_close(self, 1000, "closed")


def build_adapter() -> AngelOneProviderAdapter:
    return AngelOneProviderAdapter(
        credentials=AngelOneCredentials(
            api_key="api-key",
            client_code="CLIENT123",
            pin="1234",
            totp_secret="BASE32SECRET",
        ),
        config=AngelOneConfig(),
        smart_connect_factory=FakeSmartConnect,
        websocket_factory=FakeWebSocket,
        totp_factory=FakeTotp,
        clock=lambda: NOW,
    )


class AngelOneAdapterTests(IsolatedAsyncioTestCase):
    async def test_live_tick_is_normalized_without_fake_index_volume(self) -> None:
        adapter = build_adapter()
        await adapter.connect()
        try:
            stream = adapter.stream()
            raw = await anext(stream)
            await stream.aclose()
            registry = build_mvp_instrument_registry()
            pipeline = IngestionPipeline(
                normalizer=MarketEventNormalizer(registry),
                validator=MarketEventValidator(registry),
                ledger=InMemoryEventLedger(),
            )
            outcome = pipeline.process(raw)
        finally:
            await adapter.disconnect()

        self.assertEqual(outcome.status, IngestionStatus.STORED)
        self.assertIsNotNone(outcome.event)
        self.assertEqual(outcome.event.instrument_id, "NIFTY50_SPOT")  # type: ignore[union-attr]
        self.assertEqual(outcome.event.price, Decimal("25000.25"))  # type: ignore[union-attr]
        self.assertIsNone(outcome.event.last_quantity)  # type: ignore[union-attr]
        self.assertIsNone(outcome.event.cumulative_volume)  # type: ignore[union-attr]
        self.assertFalse(outcome.event.provider_sequence_is_contiguous)  # type: ignore[union-attr]

    async def test_historical_candle_is_finalized_and_volume_is_forced_null(self) -> None:
        adapter = build_adapter()
        await adapter.connect()
        try:
            bars = await adapter.fetch_finalized_minutes(
                from_time=datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc),
                to_time=datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc),
            )
        finally:
            await adapter.disconnect()

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].instrument_id, "NIFTY50_SPOT")
        self.assertEqual(bars[0].open, Decimal("25000"))
        self.assertEqual(bars[0].close, Decimal("25003"))
        self.assertIsNone(bars[0].volume)


class AngelOneSettingsTests(TestCase):
    def test_live_angelone_mode_requires_all_backend_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {"MARKET_DATA_MODE": "live", "MARKET_DATA_PROVIDER": "angelone"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "ANGELONE_API_KEY"):
                Settings.from_environment()

    def test_credentials_are_loaded_but_redacted_from_settings_repr(self) -> None:
        environment = {
            "APP_ENV": "test",
            "MARKET_DATA_MODE": "live",
            "MARKET_DATA_PROVIDER": "ANGELONE",
            "ANGELONE_API_KEY": "private-api-key",
            "ANGELONE_CLIENT_CODE": "CLIENT123",
            "ANGELONE_PIN": "1234",
            "ANGELONE_TOTP_SECRET": "base 32 secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_environment()

        self.assertTrue(settings.angelone_credentials_configured)
        self.assertEqual(settings.market_data_provider, "angelone")
        self.assertEqual(settings.angelone_totp_secret, "BASE32SECRET")
        self.assertNotIn("private-api-key", repr(settings))
        self.assertNotIn("CLIENT123", repr(settings))
        self.assertNotIn("1234", repr(settings))

    def test_provider_credentials_dataclass_repr_is_redacted(self) -> None:
        credentials = AngelOneCredentials(
            api_key="api-key",
            client_code="CLIENT123",
            pin="1234",
            totp_secret="BASE32SECRET",
        )
        rendered = repr(credentials)
        self.assertNotIn("api-key", rendered)
        self.assertNotIn("CLIENT123", rendered)
        self.assertNotIn("1234", rendered)
        self.assertNotIn("BASE32SECRET", rendered)

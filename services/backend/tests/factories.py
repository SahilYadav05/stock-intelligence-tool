from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from nifty_terminal.domain.enums import MarketEventType, TimestampSource
from nifty_terminal.domain.market_event import RawMarketEvent


def raw_index_event(**overrides: object) -> RawMarketEvent:
    values: dict[str, object] = {
        "provider": "replay",
        "provider_instrument_id": "NIFTY50_TEST",
        "event_type": MarketEventType.INDEX_VALUE,
        "provider_event_time": datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc),
        "provider_send_time": datetime(2026, 8, 24, 3, 45, 0, 10_000, tzinfo=timezone.utc),
        "server_arrival_time": datetime(2026, 8, 24, 3, 45, 0, 40_000, tzinfo=timezone.utc),
        "timestamp_source": TimestampSource.EXCHANGE,
        "provider_sequence": 1001,
        "provider_sequence_scope": "fixture-day-2026-08-24",
        "provider_sequence_is_contiguous": True,
        "connection_epoch": "test-session-1",
        "price": Decimal("25000.00"),
        "last_quantity": None,
        "cumulative_volume": None,
        "bid_price": None,
        "ask_price": None,
        "raw_payload": {"fixture": True, "sequence": 1001},
        "supersedes_event_id": None,
    }
    values.update(overrides)
    return RawMarketEvent(**values)  # type: ignore[arg-type]

"""Language-neutral WebSocket message envelopes."""

from __future__ import annotations

from datetime import datetime, timezone

from nifty_terminal.delivery.hub import SequencedMarketState
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.enums import ConnectionState


def market_state_message(item: SequencedMarketState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "message_type": "MARKET_STATE",
        "sequence": item.sequence,
        "sent_at": _datetime_text(item.sent_at),
        "instrument_id": item.view.snapshot.instrument_id,
        "payload": item.view.to_contract(),
    }


def current_view_message(view: MarketStateView, instrument_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "message_type": "MARKET_STATE",
        "sequence": 0,
        "sent_at": _datetime_text(datetime.now(timezone.utc)),
        "instrument_id": instrument_id,
        "payload": view.to_contract(),
    }


def status_message(
    *,
    instrument_id: str,
    data_status: ConnectionState,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "message_type": "STATUS",
        "sequence": 0,
        "sent_at": _datetime_text(datetime.now(timezone.utc)),
        "instrument_id": instrument_id,
        "payload": {
            "data_status": data_status.value,
            "reason": reason,
            "live_analysis_available": False,
        },
    }


def _datetime_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")

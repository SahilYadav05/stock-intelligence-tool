"""Provider-neutral adapter boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from nifty_terminal.domain.candle import FinalizedMinuteBarInput
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.market_event import RawMarketEvent


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    provider: str
    connection_state: ConnectionState
    observed_at: datetime
    last_event_time: datetime | None = None
    detail: str | None = None

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "provider": self.provider,
            "connection_state": self.connection_state.value,
            "observed_at": _datetime_text(self.observed_at),
            "last_event_time": _datetime_text(self.last_event_time),
            "detail": self.detail,
        }


class ProviderAdapter(ABC):
    """Contract implemented by replay and future licensed live providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def health(self) -> ProviderHealth:
        raise NotImplementedError

    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> AsyncIterator[RawMarketEvent]:
        raise NotImplementedError


@runtime_checkable
class FinalizedMinuteProvider(Protocol):
    """Optional provider capability for authoritative, finalized 1m bars."""

    async def fetch_finalized_minutes(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[FinalizedMinuteBarInput, ...]: ...


def _datetime_text(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None

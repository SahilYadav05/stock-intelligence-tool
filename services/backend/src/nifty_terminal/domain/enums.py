"""Stable enumerations shared by the canonical market-data layer."""

from enum import StrEnum


class MarketEventType(StrEnum):
    INDEX_VALUE = "INDEX_VALUE"
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    HEARTBEAT = "HEARTBEAT"
    CORRECTION = "CORRECTION"


class TimestampSource(StrEnum):
    EXCHANGE = "EXCHANGE"
    PROVIDER = "PROVIDER"
    ARRIVAL = "ARRIVAL"


class EventQualityCode(StrEnum):
    ARRIVAL_TIME_FALLBACK = "ARRIVAL_TIME_FALLBACK"
    LATE_ARRIVAL = "LATE_ARRIVAL"
    NO_PROVIDER_SEQUENCE = "NO_PROVIDER_SEQUENCE"
    PROVIDER_TIMESTAMP = "PROVIDER_TIMESTAMP"


class ValidationSeverity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


class ConnectionState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    RECOVERING = "RECOVERING"
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    STALE = "STALE"
    MARKET_CLOSED = "MARKET_CLOSED"

"""Angel One SmartAPI adapter for NIFTY 50 market data only.

The adapter deliberately exposes no order methods. Credentials and session
tokens remain inside the backend process, while every observation crossing the
boundary is converted into the provider-neutral RawMarketEvent contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from threading import Thread
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from nifty_terminal.calendar.nse import IST
from nifty_terminal.domain.candle import FinalizedMinuteBarInput
from nifty_terminal.domain.enums import ConnectionState, MarketEventType, TimestampSource
from nifty_terminal.domain.market_event import RawMarketEvent
from nifty_terminal.providers.base import ProviderAdapter, ProviderHealth
from nifty_terminal.settings import Settings


SmartConnectFactory = Callable[[str], Any]
WebSocketFactory = Callable[[str, str, str, str], Any]
TotpFactory = Callable[[str], Any]
Clock = Callable[[], datetime]
_STREAM_END = object()


class AngelOneProviderError(RuntimeError):
    """Base class for sanitized Angel One failures."""


class AngelOneAuthenticationError(AngelOneProviderError):
    """Authentication failed without exposing credentials or response tokens."""


class AngelOneConnectionError(AngelOneProviderError):
    """The live WebSocket could not become ready."""


class AngelOneDataError(AngelOneProviderError):
    """Provider data was missing or could not be represented safely."""


@dataclass(frozen=True, slots=True)
class AngelOneCredentials:
    api_key: str = field(repr=False)
    client_code: str = field(repr=False)
    pin: str = field(repr=False)
    totp_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("api_key", "client_code", "pin", "totp_secret"):
            if not getattr(self, name).strip():
                raise ValueError(f"Angel One {name} cannot be empty")


@dataclass(frozen=True, slots=True)
class AngelOneConfig:
    websocket_token: str = "99926000"
    historical_token: str = "99926000"
    exchange: str = "NSE"
    websocket_exchange_type: int = 1
    price_scale: int = 100
    connect_timeout_seconds: int = 20
    queue_capacity: int = 4_096
    correlation_id: str = "nifty-intelligence-nifty50"

    def __post_init__(self) -> None:
        if not self.websocket_token.isascii() or not self.websocket_token.isdigit():
            raise ValueError("Angel One WebSocket token must contain ASCII digits only")
        if not self.historical_token.isascii() or not self.historical_token.isdigit():
            raise ValueError("Angel One historical token must contain ASCII digits only")
        if self.websocket_exchange_type < 1:
            raise ValueError("Angel One exchange type must be positive")
        if self.price_scale < 1:
            raise ValueError("Angel One price scale must be positive")
        if self.connect_timeout_seconds < 1:
            raise ValueError("Angel One connection timeout must be positive")
        if self.queue_capacity < 1:
            raise ValueError("Angel One stream queue capacity must be positive")

    @property
    def provider_instrument_id(self) -> str:
        return f"{self.exchange}:{self.websocket_token}"


class AngelOneProviderAdapter(ProviderAdapter):
    """Thread-to-async bridge around Angel One's official Python SDK."""

    def __init__(
        self,
        *,
        credentials: AngelOneCredentials,
        config: AngelOneConfig | None = None,
        smart_connect_factory: SmartConnectFactory | None = None,
        websocket_factory: WebSocketFactory | None = None,
        totp_factory: TotpFactory | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._credentials = credentials
        self._config = config or AngelOneConfig()
        self._smart_connect_factory = smart_connect_factory
        self._websocket_factory = websocket_factory
        self._totp_factory = totp_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state = ConnectionState.DISCONNECTED
        self._detail: str | None = "Angel One adapter has not connected."
        self._last_event_time: datetime | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[RawMarketEvent | object] = asyncio.Queue(
            maxsize=self._config.queue_capacity
        )
        self._connect_future: asyncio.Future[None] | None = None
        self._socket_thread: Thread | None = None
        self._smart_client: Any = None
        self._websocket: Any = None
        self._connection_epoch = ""
        self._stop_requested = False

    @property
    def provider_name(self) -> str:
        return "angelone"

    @property
    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider_name,
            connection_state=self._state,
            observed_at=_as_utc(self._clock(), "clock"),
            last_event_time=self._last_event_time,
            detail=self._detail,
        )

    async def connect(self) -> None:
        if self._state is not ConnectionState.DISCONNECTED:
            raise AngelOneConnectionError("Angel One adapter is already active.")

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._config.queue_capacity)
        self._connect_future = self._loop.create_future()
        self._connection_epoch = str(uuid4())
        self._stop_requested = False
        self._state = ConnectionState.CONNECTING
        self._detail = "Authenticating with Angel One."

        smart_factory, websocket_factory, totp_factory = self._resolve_dependencies()
        try:
            totp = totp_factory(self._credentials.totp_secret).now()
        except Exception as error:
            self._state = ConnectionState.DISCONNECTED
            raise AngelOneAuthenticationError(
                "ANGELONE_TOTP_SECRET is not a valid SmartAPI Base32 TOTP seed."
            ) from error

        try:
            self._smart_client = await asyncio.to_thread(
                smart_factory,
                self._credentials.api_key,
            )
            session = await asyncio.to_thread(
                self._smart_client.generateSession,
                self._credentials.client_code,
                self._credentials.pin,
                totp,
            )
            auth_token, refresh_token = _validated_session(session)
            feed_token = await asyncio.to_thread(self._smart_client.getfeedToken)
            if not isinstance(feed_token, str) or not feed_token.strip():
                raise AngelOneAuthenticationError(
                    "Angel One authenticated but did not return a feed token."
                )
            self._websocket = websocket_factory(
                auth_token,
                self._credentials.api_key,
                self._credentials.client_code,
                feed_token,
            )
            self._install_callbacks()
            self._socket_thread = Thread(
                target=self._run_socket,
                name="angelone-market-websocket",
                daemon=True,
            )
            self._socket_thread.start()
            await asyncio.wait_for(
                asyncio.shield(self._connect_future),
                timeout=self._config.connect_timeout_seconds,
            )
            # Keep the refresh token inside the provider object only so a future
            # controlled refresh can be added without exposing it downstream.
            _ = refresh_token
        except TimeoutError as error:
            await self.disconnect()
            raise AngelOneConnectionError(
                "Angel One WebSocket did not open before the configured timeout."
            ) from error
        except AngelOneProviderError:
            await self.disconnect()
            raise
        except Exception as error:
            await self.disconnect()
            raise AngelOneConnectionError(
                f"Angel One setup failed safely ({type(error).__name__})."
            ) from error

    async def disconnect(self) -> None:
        self._stop_requested = True
        websocket = self._websocket
        if websocket is not None:
            try:
                # smartapi-python 1.5.5 may route an intentional close through
                # its retry callback. Disable that callback before closing so a
                # clean backend shutdown cannot consume another connection slot.
                sdk_wsapp = getattr(websocket, "wsapp", None)
                if sdk_wsapp is not None:
                    sdk_wsapp.on_error = lambda *_: None
                    sdk_wsapp.on_close = lambda *_: None
                if hasattr(websocket, "RESUBSCRIBE_FLAG"):
                    websocket.RESUBSCRIBE_FLAG = False
                if hasattr(websocket, "DISCONNECT_FLAG"):
                    websocket.DISCONNECT_FLAG = True
                await asyncio.to_thread(websocket.close_connection)
            except Exception:
                pass
        thread = self._socket_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 2.0)
        self._state = ConnectionState.DISCONNECTED
        self._detail = "Angel One adapter disconnected."
        self._enqueue(_STREAM_END)
        self._websocket = None
        self._smart_client = None
        self._socket_thread = None

    async def stream(self) -> AsyncIterator[RawMarketEvent]:
        if self._state not in {
            ConnectionState.LIVE,
            ConnectionState.RECOVERING,
            ConnectionState.DELAYED,
            ConnectionState.STALE,
        }:
            raise AngelOneConnectionError(
                "Angel One adapter must be connected before streaming."
            )
        while True:
            item = await self._queue.get()
            if item is _STREAM_END:
                return
            if not isinstance(item, RawMarketEvent):
                raise AngelOneDataError("Unexpected object reached the market-data queue.")
            yield item

    async def fetch_finalized_minutes(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[FinalizedMinuteBarInput, ...]:
        start = _as_utc(from_time, "from_time")
        end = _as_utc(to_time, "to_time")
        if start >= end:
            raise ValueError("from_time must be earlier than to_time")
        if self._smart_client is None:
            raise AngelOneConnectionError(
                "Angel One must be authenticated before requesting historical candles."
            )

        request = {
            "exchange": self._config.exchange,
            "symboltoken": self._config.historical_token,
            "interval": "ONE_MINUTE",
            "fromdate": start.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
            "todate": end.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
        }
        try:
            response = await asyncio.to_thread(self._smart_client.getCandleData, request)
        except Exception as error:
            raise AngelOneDataError(
                f"Angel One historical request failed safely ({type(error).__name__})."
            ) from error
        rows = _validated_candle_rows(response)
        finalized_at = _as_utc(self._clock(), "clock")
        bars = tuple(
            _historical_row_to_minute(
                row,
                config=self._config,
                finalized_at=finalized_at,
            )
            for row in rows
        )
        selected = tuple(
            item
            for item in bars
            if start <= item.opens_at and item.closes_at <= end
        )
        _validate_unique_history(selected)
        return tuple(sorted(selected, key=lambda item: item.opens_at))

    def _resolve_dependencies(
        self,
    ) -> tuple[SmartConnectFactory, WebSocketFactory, TotpFactory]:
        smart_factory = self._smart_connect_factory
        websocket_factory = self._websocket_factory
        totp_factory = self._totp_factory
        try:
            if smart_factory is None:
                from SmartApi import SmartConnect

                smart_factory = SmartConnect
            if websocket_factory is None:
                from SmartApi.smartWebSocketV2 import SmartWebSocketV2

                websocket_factory = SmartWebSocketV2
            if totp_factory is None:
                import pyotp

                totp_factory = pyotp.TOTP
        except ImportError as error:
            raise AngelOneConnectionError(
                "Angel One dependencies are missing. Install the provider-angelone extra."
            ) from error
        return smart_factory, websocket_factory, totp_factory

    def _install_callbacks(self) -> None:
        self._websocket.on_open = self._on_open
        self._websocket.on_data = self._on_data
        self._websocket.on_error = self._on_error
        self._websocket.on_close = self._on_close
        if hasattr(self._websocket, "on_control_message"):
            self._websocket.on_control_message = self._on_control_message

    def _run_socket(self) -> None:
        try:
            self._websocket.connect()
        except Exception as error:
            self._dispatch(self._mark_connection_failure, type(error).__name__)

    def _on_open(self, *_: object) -> None:
        try:
            self._websocket.subscribe(
                self._config.correlation_id,
                1,
                [
                    {
                        "exchangeType": self._config.websocket_exchange_type,
                        "tokens": [self._config.websocket_token],
                    }
                ],
            )
        except Exception as error:
            self._dispatch(self._mark_connection_failure, type(error).__name__)
            return
        self._dispatch(self._mark_open)

    def _on_data(self, _: object, message: object) -> None:
        arrival = _as_utc(self._clock(), "clock")
        try:
            event = _tick_to_raw_event(
                message,
                config=self._config,
                arrival=arrival,
                connection_epoch=self._connection_epoch,
            )
        except AngelOneDataError as error:
            self._dispatch(self._mark_bad_tick, str(error))
            return
        self._dispatch(self._accept_event, event)

    def _on_error(self, _: object, error: object) -> None:
        self._dispatch(self._mark_socket_error, type(error).__name__)

    def _on_close(self, *_: object) -> None:
        self._dispatch(self._mark_closed)

    def _on_control_message(self, *_: object) -> None:
        return

    def _dispatch(self, callback: Callable[..., None], *args: object) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(callback, *args)

    def _mark_open(self) -> None:
        self._state = ConnectionState.LIVE
        self._detail = "Angel One WebSocket subscribed to NIFTY 50."
        if self._connect_future is not None and not self._connect_future.done():
            self._connect_future.set_result(None)

    def _mark_connection_failure(self, error_type: str) -> None:
        self._state = ConnectionState.DISCONNECTED
        self._detail = f"Angel One WebSocket setup failed ({error_type})."
        if self._connect_future is not None and not self._connect_future.done():
            self._connect_future.set_exception(AngelOneConnectionError(self._detail))

    def _mark_socket_error(self, error_type: str) -> None:
        self._state = ConnectionState.RECOVERING
        self._detail = f"Angel One WebSocket is recovering ({error_type})."
        if self._connect_future is not None and not self._connect_future.done():
            self._connect_future.set_exception(AngelOneConnectionError(self._detail))

    def _mark_closed(self) -> None:
        if self._stop_requested:
            self._state = ConnectionState.DISCONNECTED
            self._detail = "Angel One WebSocket closed by the backend."
            self._enqueue(_STREAM_END)
        else:
            self._state = ConnectionState.RECOVERING
            self._detail = "Angel One WebSocket closed unexpectedly; SDK recovery is pending."

    def _mark_bad_tick(self, detail: str) -> None:
        self._detail = f"Angel One rejected a malformed tick: {detail}"

    def _accept_event(self, event: RawMarketEvent) -> None:
        self._last_event_time = event.provider_event_time or event.server_arrival_time
        self._state = ConnectionState.LIVE
        self._detail = "Angel One NIFTY 50 observations are arriving."
        self._enqueue(event)

    def _enqueue(self, item: RawMarketEvent | object) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._detail = "Angel One stream queue overflowed; oldest observation was dropped."
            self._state = ConnectionState.DELAYED
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._state = ConnectionState.STALE
            self._detail = "Angel One stream queue remained full; data is stale."


def build_angelone_adapter(settings: Settings) -> AngelOneProviderAdapter:
    """Construct the single provider adapter from validated server-only settings."""

    if settings.market_data_mode != "live" or settings.market_data_provider != "angelone":
        raise ValueError("Angel One adapter requires MARKET_DATA_MODE=live and provider=angelone")
    if not settings.angelone_credentials_configured:
        raise ValueError("Angel One credentials are incomplete")
    return AngelOneProviderAdapter(
        credentials=AngelOneCredentials(
            api_key=settings.angelone_api_key or "",
            client_code=settings.angelone_client_code or "",
            pin=settings.angelone_pin or "",
            totp_secret=settings.angelone_totp_secret or "",
        ),
        config=AngelOneConfig(
            websocket_token=settings.angelone_nifty_websocket_token,
            historical_token=settings.angelone_nifty_historical_token,
            websocket_exchange_type=settings.angelone_websocket_exchange_type,
            price_scale=settings.angelone_price_scale,
            connect_timeout_seconds=settings.angelone_connect_timeout_seconds,
            queue_capacity=settings.angelone_stream_queue_capacity,
        ),
    )


def _validated_session(response: object) -> tuple[str, str]:
    if not isinstance(response, Mapping):
        raise AngelOneAuthenticationError("Angel One returned an invalid login response.")
    if response.get("status") is not True:
        code = str(response.get("errorcode") or "UNKNOWN")
        message = str(response.get("message") or "authentication rejected")
        raise AngelOneAuthenticationError(
            f"Angel One authentication failed [{code}]: {message}"
        )
    data = response.get("data")
    if not isinstance(data, Mapping):
        raise AngelOneAuthenticationError("Angel One login response did not contain session data.")
    auth = data.get("jwtToken")
    refresh = data.get("refreshToken")
    if not isinstance(auth, str) or not auth or not isinstance(refresh, str) or not refresh:
        raise AngelOneAuthenticationError("Angel One login response omitted required tokens.")
    return auth, refresh


def _tick_to_raw_event(
    message: object,
    *,
    config: AngelOneConfig,
    arrival: datetime,
    connection_epoch: str,
) -> RawMarketEvent:
    if not isinstance(message, Mapping):
        raise AngelOneDataError("tick is not an object")
    token = str(_first(message, "token", default="")).replace("\x00", "").strip()
    if token != config.websocket_token:
        raise AngelOneDataError(f"unexpected token {token!r}")
    price = _scaled_decimal(
        _required(message, "last_traded_price", "lastTradedPrice", "ltp"),
        config.price_scale,
        "last_traded_price",
    )
    sequence_value = _first(message, "sequence_number", "sequenceNumber")
    sequence = int(sequence_value) if sequence_value is not None else None
    timestamp_value = _first(message, "exchange_timestamp", "exchangeTimeStamp")
    event_time = _epoch_milliseconds(timestamp_value) if timestamp_value is not None else None
    return RawMarketEvent(
        provider="angelone",
        provider_instrument_id=config.provider_instrument_id,
        event_type=MarketEventType.INDEX_VALUE,
        server_arrival_time=arrival,
        connection_epoch=connection_epoch,
        raw_payload=_json_safe(message),
        provider_event_time=event_time,
        provider_send_time=None,
        timestamp_source=(
            TimestampSource.EXCHANGE if event_time is not None else TimestampSource.ARRIVAL
        ),
        provider_sequence=sequence,
        provider_sequence_scope=(
            f"angelone:{config.websocket_exchange_type}:{token}" if sequence is not None else None
        ),
        provider_sequence_is_contiguous=False,
        price=price,
        last_quantity=None,
        cumulative_volume=None,
        bid_price=None,
        ask_price=None,
    )


def _validated_candle_rows(response: object) -> tuple[object, ...]:
    if not isinstance(response, Mapping):
        raise AngelOneDataError("Angel One returned an invalid historical response.")
    if response.get("status") is not True:
        code = str(response.get("errorcode") or "UNKNOWN")
        message = str(response.get("message") or "historical request rejected")
        raise AngelOneDataError(f"Angel One historical request failed [{code}]: {message}")
    data = response.get("data")
    if data is None:
        return ()
    if not isinstance(data, (list, tuple)):
        raise AngelOneDataError("Angel One historical data is not a row list.")
    return tuple(data)


def _historical_row_to_minute(
    row: object,
    *,
    config: AngelOneConfig,
    finalized_at: datetime,
) -> FinalizedMinuteBarInput:
    if not isinstance(row, (list, tuple)) or len(row) < 5:
        raise AngelOneDataError("Historical candle row must contain timestamp and OHLC.")
    opens_at = _parse_provider_datetime(row[0])
    closes_at = opens_at + timedelta(minutes=1)
    values = tuple(_decimal(row[index], f"OHLC[{index}]") for index in range(1, 5))
    open_price, high, low, close = values
    if min(values) <= Decimal("0"):
        raise AngelOneDataError("Historical OHLC must be positive.")
    if low > min(open_price, close) or high < max(open_price, close) or low > high:
        raise AngelOneDataError("Historical candle violates OHLC invariants.")
    identity = json.dumps(
        {
            "provider": "angelone",
            "token": config.historical_token,
            "opens_at": opens_at.isoformat(),
            "open": format(open_price, "f"),
            "high": format(high, "f"),
            "low": format(low, "f"),
            "close": format(close, "f"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return FinalizedMinuteBarInput(
        provider_bar_id=str(uuid5(NAMESPACE_URL, identity)),
        provider="angelone",
        instrument_id="NIFTY50_SPOT",
        opens_at=opens_at,
        closes_at=closes_at,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=None,
        provider_revision=1,
        finalized_at=finalized_at,
        source_watermark=f"angelone:{config.historical_token}:{fingerprint}",
    )


def _validate_unique_history(bars: tuple[FinalizedMinuteBarInput, ...]) -> None:
    seen: dict[datetime, str] = {}
    for bar in bars:
        existing = seen.get(bar.opens_at)
        if existing is not None and existing != bar.provider_bar_id:
            raise AngelOneDataError(
                "Angel One returned conflicting historical rows for one minute."
            )
        seen[bar.opens_at] = bar.provider_bar_id


def _parse_provider_datetime(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise AngelOneDataError("Historical timestamp is not ISO-8601.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AngelOneDataError("Historical timestamp must include an explicit offset.")
    return parsed.astimezone(timezone.utc)


def _epoch_milliseconds(value: object) -> datetime | None:
    try:
        milliseconds = int(str(value))
    except (TypeError, ValueError) as error:
        raise AngelOneDataError("Exchange timestamp is not an integer.") from error
    if milliseconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1_000, timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise AngelOneDataError("Exchange timestamp is outside the supported range.") from error


def _scaled_decimal(value: object, scale: int, field_name: str) -> Decimal:
    return _decimal(value, field_name) / Decimal(scale)


def _decimal(value: object, field_name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise AngelOneDataError(f"{field_name} is not numeric.") from error
    if not parsed.is_finite():
        raise AngelOneDataError(f"{field_name} must be finite.")
    return parsed


def _required(message: Mapping[object, object], *names: str) -> object:
    value = _first(message, *names)
    if value is None:
        raise AngelOneDataError(f"tick omitted {names[0]}")
    return value


def _first(
    message: Mapping[object, object],
    *names: str,
    default: object | None = None,
) -> object | None:
    for name in names:
        if name in message:
            return message[name]
    return default


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)

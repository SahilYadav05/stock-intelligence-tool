"""Server-side settings for the backend foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


ALLOWED_ENVIRONMENTS = frozenset({"development", "test", "production"})
ALLOWED_MARKET_DATA_MODES = frozenset({"replay", "live"})
ALLOWED_API_AUTH_MODES = frozenset({"disabled", "bearer"})
DEFAULT_API_ALLOWED_ORIGINS = (
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:5173",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings that are safe to construct without external services."""

    app_name: str
    environment: str
    log_level: str
    market_data_mode: str
    market_data_provider: str | None
    api_allowed_origins: tuple[str, ...] = DEFAULT_API_ALLOWED_ORIGINS
    api_auth_mode: str = "disabled"
    api_auth_token: str | None = field(default=None, repr=False)
    request_body_limit_bytes: int = 65_536
    requests_per_minute: int = 120
    websocket_connection_limit: int = 5
    release_manifest_path: Path | None = None
    model_artifact_path: Path | None = None
    calibration_artifact_path: Path | None = None
    live_signal_kill_switch: bool = True
    angelone_api_key: str | None = field(default=None, repr=False)
    angelone_client_code: str | None = field(default=None, repr=False)
    angelone_pin: str | None = field(default=None, repr=False)
    angelone_totp_secret: str | None = field(default=None, repr=False)
    angelone_nifty_websocket_token: str = "99926000"
    angelone_nifty_historical_token: str = "99926000"
    angelone_websocket_exchange_type: int = 1
    angelone_price_scale: int = 100
    angelone_connect_timeout_seconds: int = 20
    angelone_stream_queue_capacity: int = 4_096
    live_history_lookback_days: int = 14
    live_history_recovery_minutes: int = 15
    live_history_poll_seconds: int = 10
    live_minute_finalization_delay_seconds: int = 5
    live_tick_fresh_seconds: int = 3
    live_tick_stale_seconds: int = 15
    live_chart_publish_interval_milliseconds: int = 250
    live_chart_history_primary_limit: int = 750
    live_chart_history_context_limit: int = 250
    live_chart_history_hourly_limit: int = 120
    shadow_mode_enabled: bool = False
    shadow_runtime_manifest_path: Path | None = None
    shadow_ledger_path: Path = Path("data/shadow-ledger.sqlite3")

    def __post_init__(self) -> None:
        if self.live_tick_fresh_seconds >= self.live_tick_stale_seconds:
            raise ValueError(
                "LIVE_TICK_FRESH_SECONDS must be lower than LIVE_TICK_STALE_SECONDS"
            )
        if self.shadow_mode_enabled and self.shadow_runtime_manifest_path is None:
            raise ValueError(
                "SHADOW_RUNTIME_MANIFEST_PATH is required when SHADOW_MODE_ENABLED=true"
            )
        if self.shadow_mode_enabled and not self.live_signal_kill_switch:
            raise ValueError("Shadow mode requires LIVE_SIGNAL_KILL_SWITCH=true")

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.getenv("APP_ENV", "development").strip().lower()
        if environment not in ALLOWED_ENVIRONMENTS:
            allowed = ", ".join(sorted(ALLOWED_ENVIRONMENTS))
            raise ValueError(f"APP_ENV must be one of: {allowed}")

        provider_text = os.getenv("MARKET_DATA_PROVIDER", "").strip().lower()
        provider = provider_text or None
        market_data_mode = os.getenv("MARKET_DATA_MODE", "replay").strip().lower()
        if market_data_mode not in ALLOWED_MARKET_DATA_MODES:
            allowed = ", ".join(sorted(ALLOWED_MARKET_DATA_MODES))
            raise ValueError(f"MARKET_DATA_MODE must be one of: {allowed}")
        if market_data_mode == "live" and provider is None:
            raise ValueError("MARKET_DATA_PROVIDER is required when MARKET_DATA_MODE=live")

        origin_text = os.getenv("API_ALLOWED_ORIGINS", "").strip()
        origins = tuple(
            item.strip().rstrip("/") for item in origin_text.split(",") if item.strip()
        ) or DEFAULT_API_ALLOWED_ORIGINS
        if "*" in origins:
            raise ValueError("API_ALLOWED_ORIGINS cannot contain a wildcard")

        api_auth_mode = os.getenv("API_AUTH_MODE", "disabled").strip().lower()
        if api_auth_mode not in ALLOWED_API_AUTH_MODES:
            allowed = ", ".join(sorted(ALLOWED_API_AUTH_MODES))
            raise ValueError(f"API_AUTH_MODE must be one of: {allowed}")
        api_auth_token = os.getenv("API_AUTH_TOKEN", "").strip() or None
        if api_auth_mode == "bearer" and (api_auth_token is None or len(api_auth_token) < 32):
            raise ValueError("API_AUTH_TOKEN must contain at least 32 characters in bearer mode")
        if environment == "production":
            if api_auth_mode != "bearer":
                raise ValueError("Production requires API_AUTH_MODE=bearer")
            if any(not origin.startswith("https://") for origin in origins):
                raise ValueError("Production API_ALLOWED_ORIGINS must use HTTPS")

        request_body_limit_bytes = _bounded_int("REQUEST_BODY_LIMIT_BYTES", 65_536, 1_024, 1_048_576)
        requests_per_minute = _bounded_int("REQUESTS_PER_MINUTE", 120, 10, 10_000)
        websocket_connection_limit = _bounded_int("WEBSOCKET_CONNECTION_LIMIT", 5, 1, 100)
        release_path_text = os.getenv("RELEASE_MANIFEST_PATH", "").strip()
        model_path_text = os.getenv("MODEL_ARTIFACT_PATH", "").strip()
        calibration_path_text = os.getenv("CALIBRATION_ARTIFACT_PATH", "").strip()
        shadow_manifest_text = os.getenv("SHADOW_RUNTIME_MANIFEST_PATH", "").strip()
        shadow_ledger_text = os.getenv(
            "SHADOW_LEDGER_PATH", "data/shadow-ledger.sqlite3"
        ).strip()
        angelone_api_key = os.getenv("ANGELONE_API_KEY", "").strip() or None
        angelone_client_code = os.getenv("ANGELONE_CLIENT_CODE", "").strip() or None
        angelone_pin = os.getenv("ANGELONE_PIN", "").strip() or None
        angelone_totp_secret = (
            os.getenv("ANGELONE_TOTP_SECRET", "").replace(" ", "").strip().upper() or None
        )
        if market_data_mode == "live" and provider == "angelone":
            missing = [
                name
                for name, value in (
                    ("ANGELONE_API_KEY", angelone_api_key),
                    ("ANGELONE_CLIENT_CODE", angelone_client_code),
                    ("ANGELONE_PIN", angelone_pin),
                    ("ANGELONE_TOTP_SECRET", angelone_totp_secret),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Angel One live mode requires backend-only credentials: "
                    + ", ".join(missing)
                )

        websocket_token = _digits(
            "ANGELONE_NIFTY_WEBSOCKET_TOKEN",
            "99926000",
        )
        historical_token = _digits(
            "ANGELONE_NIFTY_HISTORICAL_TOKEN",
            "99926000",
        )

        return cls(
            app_name=os.getenv("APP_NAME", "NIFTY Intelligence Terminal").strip(),
            environment=environment,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            market_data_mode=market_data_mode,
            market_data_provider=provider,
            api_allowed_origins=origins,
            api_auth_mode=api_auth_mode,
            api_auth_token=api_auth_token,
            request_body_limit_bytes=request_body_limit_bytes,
            requests_per_minute=requests_per_minute,
            websocket_connection_limit=websocket_connection_limit,
            release_manifest_path=Path(release_path_text) if release_path_text else None,
            model_artifact_path=Path(model_path_text) if model_path_text else None,
            calibration_artifact_path=(
                Path(calibration_path_text) if calibration_path_text else None
            ),
            live_signal_kill_switch=_environment_bool("LIVE_SIGNAL_KILL_SWITCH", True),
            angelone_api_key=angelone_api_key,
            angelone_client_code=angelone_client_code,
            angelone_pin=angelone_pin,
            angelone_totp_secret=angelone_totp_secret,
            angelone_nifty_websocket_token=websocket_token,
            angelone_nifty_historical_token=historical_token,
            angelone_websocket_exchange_type=_bounded_int(
                "ANGELONE_WEBSOCKET_EXCHANGE_TYPE", 1, 1, 20
            ),
            angelone_price_scale=_bounded_int("ANGELONE_PRICE_SCALE", 100, 1, 1_000_000),
            angelone_connect_timeout_seconds=_bounded_int(
                "ANGELONE_CONNECT_TIMEOUT_SECONDS", 20, 5, 120
            ),
            angelone_stream_queue_capacity=_bounded_int(
                "ANGELONE_STREAM_QUEUE_CAPACITY", 4_096, 128, 100_000
            ),
            live_history_lookback_days=_bounded_int(
                "LIVE_HISTORY_LOOKBACK_DAYS", 14, 2, 30
            ),
            live_history_recovery_minutes=_bounded_int(
                "LIVE_HISTORY_RECOVERY_MINUTES", 15, 5, 120
            ),
            live_history_poll_seconds=_bounded_int(
                "LIVE_HISTORY_POLL_SECONDS", 10, 5, 60
            ),
            live_minute_finalization_delay_seconds=_bounded_int(
                "LIVE_MINUTE_FINALIZATION_DELAY_SECONDS", 5, 2, 30
            ),
            live_tick_fresh_seconds=_bounded_int(
                "LIVE_TICK_FRESH_SECONDS", 3, 1, 10
            ),
            live_tick_stale_seconds=_bounded_int(
                "LIVE_TICK_STALE_SECONDS", 15, 5, 120
            ),
            live_chart_publish_interval_milliseconds=_bounded_int(
                "LIVE_CHART_PUBLISH_INTERVAL_MILLISECONDS", 250, 100, 2_000
            ),
            live_chart_history_primary_limit=_bounded_int(
                "LIVE_CHART_HISTORY_PRIMARY_LIMIT", 750, 120, 2_250
            ),
            live_chart_history_context_limit=_bounded_int(
                "LIVE_CHART_HISTORY_CONTEXT_LIMIT", 250, 64, 750
            ),
            live_chart_history_hourly_limit=_bounded_int(
                "LIVE_CHART_HISTORY_HOURLY_LIMIT", 120, 32, 365
            ),
            shadow_mode_enabled=_environment_bool("SHADOW_MODE_ENABLED", False),
            shadow_runtime_manifest_path=(
                Path(shadow_manifest_text) if shadow_manifest_text else None
            ),
            shadow_ledger_path=Path(shadow_ledger_text),
        )

    @property
    def live_analysis_available(self) -> bool:
        """Availability is decided by the release gate, never configuration alone."""

        return False

    @property
    def angelone_credentials_configured(self) -> bool:
        return all(
            (
                self.angelone_api_key,
                self.angelone_client_code,
                self.angelone_pin,
                self.angelone_totp_secret,
            )
        )


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _digits(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value or not value.isascii() or not value.isdigit():
        raise ValueError(f"{name} must contain ASCII digits only")
    return value

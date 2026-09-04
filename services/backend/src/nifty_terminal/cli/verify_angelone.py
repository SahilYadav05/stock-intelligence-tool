"""Credential-safe Angel One connectivity and canonical-data verifier."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.ledger import InMemoryEventLedger
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer
from nifty_terminal.ingestion.pipeline import IngestionPipeline, IngestionStatus
from nifty_terminal.ingestion.validator import MarketEventValidator
from nifty_terminal.providers.angelone import (
    AngelOneConfig,
    AngelOneCredentials,
    AngelOneProviderAdapter,
    AngelOneProviderError,
)
from nifty_terminal.settings import Settings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Angel One login, historical candles, WebSocket, and normalization."
    )
    parser.add_argument("--env-file", default=".env", help="Backend environment file")
    parser.add_argument(
        "--history-days",
        type=int,
        default=7,
        help="Calendar days requested from the 1m historical endpoint",
    )
    parser.add_argument(
        "--tick-timeout",
        type=int,
        default=25,
        help="Seconds to wait for a NIFTY 50 tick while NSE is open",
    )
    return parser.parse_args()


def _adapter(settings: Settings) -> AngelOneProviderAdapter:
    if settings.market_data_mode != "live" or settings.market_data_provider != "angelone":
        raise ValueError(
            "Set MARKET_DATA_MODE=live and MARKET_DATA_PROVIDER=angelone in .env first."
        )
    if not settings.live_signal_kill_switch:
        raise ValueError("Keep LIVE_SIGNAL_KILL_SWITCH=true during provider validation.")
    if not settings.angelone_credentials_configured:
        raise ValueError("Angel One credentials are incomplete in .env.")

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


async def _verify(args: argparse.Namespace) -> int:
    env_path = Path(args.env_file)
    if not env_path.is_file():
        raise ValueError(f"Environment file not found: {env_path}")
    if not 1 <= args.history_days <= 30:
        raise ValueError("--history-days must be between 1 and 30")
    if not 5 <= args.tick_timeout <= 120:
        raise ValueError("--tick-timeout must be between 5 and 120")

    load_dotenv(env_path, override=False)
    settings = Settings.from_environment()
    adapter = _adapter(settings)
    now = datetime.now(timezone.utc)

    print("Angel One Step 11 verifier")
    print("- instrument: NIFTY50_SPOT")
    print(f"- WebSocket token: {settings.angelone_nifty_websocket_token}")
    print(f"- historical token: {settings.angelone_nifty_historical_token}")
    print("- NIFTY spot volume: disabled/null")
    print("- automatic trading: disabled")
    print("- live signal kill switch: ACTIVE")

    await adapter.connect()
    try:
        print(f"- authentication/WebSocket: {adapter.health.connection_state.value}")
        bars = await adapter.fetch_finalized_minutes(
            from_time=now - timedelta(days=args.history_days),
            to_time=now,
        )
        if not bars:
            raise RuntimeError(
                "Angel One returned no finalized 1m NIFTY candles for the requested window."
            )
        calendar = NseSessionCalendar()
        continuous_bars = tuple(
            item
            for item in bars
            if calendar.market_phase(item.opens_at).value == "CONTINUOUS_TRADING"
        )
        auction_observations = tuple(
            item for item in bars if calendar.market_phase(item.opens_at).is_closing_auction
        )
        if not continuous_bars:
            raise RuntimeError("Angel One returned no continuous-session NIFTY candles.")
        latest = continuous_bars[-1]
        if latest.volume is not None:
            raise RuntimeError("NIFTY spot volume must remain null.")

        candle_engine = CandleEngine(
            calendar=NseSessionCalendar(),
            registry=build_mvp_instrument_registry(),
            store=InMemoryCandleStore(),
        )
        canonical_minute = candle_engine.ingest_finalized_minute(latest).minute_candle
        print(f"- historical provider observations: {len(bars)}")
        print(f"- continuous finalized 1m candles: {len(continuous_bars)}")
        print(f"- separate closing-auction observations: {len(auction_observations)}")
        print(
            "- latest finalized candle: "
            f"{canonical_minute.opens_at.astimezone(IST).isoformat()} "
            f"O={canonical_minute.open} H={canonical_minute.high} "
            f"L={canonical_minute.low} C={canonical_minute.close} V=null"
        )

        if calendar.market_phase(now).is_closing_auction:
            print("- live tick: skipped because NSE closing auction is active")
        elif calendar.session_containing(now) is None:
            print("- live tick: skipped because continuous NSE trading is closed")
        else:
            stream = adapter.stream()
            try:
                raw = await asyncio.wait_for(anext(stream), timeout=args.tick_timeout)
            finally:
                await stream.aclose()
            registry = build_mvp_instrument_registry()
            ledger = InMemoryEventLedger()
            pipeline = IngestionPipeline(
                normalizer=MarketEventNormalizer(registry),
                validator=MarketEventValidator(registry),
                ledger=ledger,
            )
            outcome = pipeline.process(raw)
            if outcome.status is not IngestionStatus.STORED or outcome.event is None:
                raise RuntimeError(
                    f"Live tick failed canonical ingestion: {outcome.status.value}"
                )
            event = outcome.event
            print(
                "- canonical live tick: "
                f"{event.instrument_id} price={event.price} "
                f"time={event.normalized_event_time.astimezone(IST).isoformat()} V=null"
            )

        print("RESULT: ANGEL ONE PROVIDER VERIFICATION PASSED")
        print("Signals remain disabled; this command only validates market data.")
        return 0
    finally:
        await adapter.disconnect()


def main() -> int:
    args = _arguments()
    try:
        return asyncio.run(_verify(args))
    except (AngelOneProviderError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"RESULT: VERIFICATION FAILED: {error}")
        print("No signal was generated and no order endpoint was called.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

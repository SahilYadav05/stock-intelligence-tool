"""Acquire canonical Bank Nifty and India VIX context for Step 18C."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys
import hashlib
import urllib.request

from dotenv import load_dotenv

from nifty_terminal.cli.run_real_data_research import DEFAULT_CALENDAR, _require_live_signal_kill_switch
from nifty_terminal.context.angelone_history import acquire_instrument
from nifty_terminal.context.bundle import ContextBundle, bundle_sha256, write_bundle
from nifty_terminal.history.calendar_loader import load_nse_calendar, validate_calendar_coverage
from nifty_terminal.settings import Settings


SPECS = (
    ("BANKNIFTY_SPOT", "ANGELONE_BANKNIFTY_HISTORICAL_TOKEN", "99926009", "INDEX"),
    ("INDIA_VIX_SPOT", "ANGELONE_INDIA_VIX_HISTORICAL_TOKEN", "99926017", "VOLATILITY_INDEX"),
)
DEFAULT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--to-date", type=date.fromisoformat)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--request-delay-ms", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/context/angelone"))
    parser.add_argument("--instrument-master-url", default=DEFAULT_MASTER_URL)
    args = parser.parse_args()
    _require_live_signal_kill_switch(args.env_file)
    load_dotenv(args.env_file, override=False)
    settings = Settings.from_environment()
    if not settings.angelone_credentials_configured:
        raise SystemExit("Angel One credentials are incomplete")
    metadata = json.loads(args.calendar_json.read_text(encoding="utf-8"))
    through = args.to_date or date.fromisoformat(str(metadata["verified_through"]))
    validate_calendar_coverage(metadata, starts_on=args.from_date, ends_on=through)
    calendar = load_nse_calendar(args.calendar_json)
    try:
        from SmartApi import SmartConnect
        import pyotp
    except ImportError as error:
        raise SystemExit("Angel One Python dependencies are not installed") from error
    print("Step 18C canonical cross-market context acquisition", file=sys.stderr)
    print(f"- range: {args.from_date} through {through} IST", file=sys.stderr)
    print("- required: BANKNIFTY_SPOT and INDIA_VIX_SPOT", file=sys.stderr)
    print("- finalized 1m history; index volume forced null", file=sys.stderr)
    print("- provider tokens stay configurable and server-side", file=sys.stderr)
    configured_tokens = {
        instrument_id: os.getenv(env_name, default).strip()
        for instrument_id, env_name, default, _ in SPECS
    }
    master_sha256, verified_master_rows = _verify_instrument_master(
        args.instrument_master_url, configured_tokens
    )
    print(f"- instrument master verified; SHA-256 {master_sha256}", file=sys.stderr)
    client = SmartConnect(settings.angelone_api_key or "")
    session = client.generateSession(
        settings.angelone_client_code or "",
        settings.angelone_pin or "",
        pyotp.TOTP(settings.angelone_totp_secret or "").now(),
    )
    if not isinstance(session, dict) or session.get("status") is not True:
        raise SystemExit("Angel One authentication failed safely")
    instruments = []
    for instrument_id, env_name, default, asset_kind in SPECS:
        token = configured_tokens[instrument_id]
        instruments.append(acquire_instrument(
            smart_client=client,
            calendar=calendar,
            instrument_id=instrument_id,
            exchange="NSE",
            token=token,
            asset_kind=asset_kind,
            from_date=args.from_date,
            through_date=through,
            chunk_days=args.chunk_days,
            request_delay_ms=args.request_delay_ms,
            progress=_progress,
        ))
    bundle = ContextBundle(
        schema_version=1,
        provider="angelone",
        requested_from=args.from_date.isoformat(),
        requested_through=through.isoformat(),
        acquired_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        instruments=tuple(instruments),
        source_notes=(
            "Index volume is unavailable and stored as null.",
            "Only explicitly verified NSE continuous-session minutes are retained.",
            "This bundle is research-only and does not enable signals.",
            f"Instrument master SHA-256: {master_sha256}",
            "Verified instrument rows: " + json.dumps(verified_master_rows, sort_keys=True),
        ),
    )
    computed_digest = bundle_sha256(bundle)
    digest_name = f"context-{args.from_date}-{through}-{computed_digest[:12]}.json.gz"
    path = args.output_dir / digest_name
    digest = write_bundle(path, bundle)
    blockers = []
    quality = {}
    for item in instruments:
        quality[item.instrument_id] = {
            "observed_minutes": len(item.bars),
            "expected_minutes": item.expected_minutes,
            "coverage_ratio": item.coverage_ratio,
            "excluded_out_of_session": item.excluded_out_of_session,
        }
        if item.coverage_ratio < 0.98:
            blockers.append(f"{item.instrument_id}_COVERAGE_BELOW_98_PERCENT")
    report = {
        "schema_version": 1,
        "bundle_path": str(path),
        "bundle_sha256": digest,
        "quality": quality,
        "research_ready": not blockers,
        "blockers": blockers,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("RESULT: STEP 18C CONTEXT ACQUISITION " + ("PASSED" if not blockers else "BLOCKED"))
    if blockers:
        raise SystemExit(2)


def _progress(name: str, completed: int, total: int, rows: int) -> None:
    if completed == 1 or completed == total or completed % 10 == 0:
        print(f"- {name}: chunks {completed}/{total}; unique rows {rows}", file=sys.stderr)


def _verify_instrument_master(
    url: str, configured_tokens: dict[str, str]
) -> tuple[str, dict[str, dict[str, str]]]:
    request = urllib.request.Request(url, headers={"User-Agent": "stock-intelligence-private/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    except Exception as error:
        raise SystemExit(
            "Official Angel One instrument master could not be verified; acquisition "
            f"stopped safely ({type(error).__name__}). Retry when the provider endpoint is healthy."
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit("Angel One instrument master is not valid JSON") from error
    if not isinstance(payload, list):
        raise SystemExit("Angel One instrument master did not contain an instrument list")
    expected_text = {"BANKNIFTY_SPOT": "BANKNIFTY", "INDIA_VIX_SPOT": "INDIAVIX"}
    verified = {}
    for instrument_id, token in configured_tokens.items():
        matches = [item for item in payload if isinstance(item, dict) and str(item.get("token")) == token]
        if len(matches) != 1:
            raise SystemExit(f"Instrument master did not uniquely resolve {instrument_id} token {token}")
        row = matches[0]
        identity = "".join(character for character in (
            str(row.get("name", "")) + str(row.get("symbol", ""))
        ).upper() if character.isalnum())
        if expected_text[instrument_id] not in identity or str(row.get("exch_seg", "")).upper() != "NSE":
            raise SystemExit(
                f"Configured token {token} does not resolve to expected {instrument_id}; acquisition stopped"
            )
        verified[instrument_id] = {
            "token": token,
            "name": str(row.get("name", "")),
            "symbol": str(row.get("symbol", "")),
            "exchange": str(row.get("exch_seg", "")),
            "instrument_type": str(row.get("instrumenttype", "")),
        }
    return hashlib.sha256(raw).hexdigest(), verified


if __name__ == "__main__":
    main()

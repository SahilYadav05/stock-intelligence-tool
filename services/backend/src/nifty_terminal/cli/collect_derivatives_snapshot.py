"""Collect one append-only NIFTY derivatives snapshot for future research."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import urllib.request

from dotenv import load_dotenv

from nifty_terminal.cli.acquire_angelone_context import DEFAULT_MASTER_URL
from nifty_terminal.cli.run_real_data_research import _require_live_signal_kill_switch
from nifty_terminal.derivatives.forward import (
    SQLiteDerivativesLedger,
    build_derivatives_snapshot,
    resolve_nifty_contracts,
)
from nifty_terminal.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--database", type=Path, default=Path("data/derivatives-forward.sqlite3"))
    parser.add_argument("--instrument-master-url", default=DEFAULT_MASTER_URL)
    args = parser.parse_args()
    _require_live_signal_kill_switch(args.env_file)
    load_dotenv(args.env_file, override=False)
    settings = Settings.from_environment()
    if not settings.angelone_credentials_configured:
        raise SystemExit("Angel One credentials are incomplete")
    try:
        from SmartApi import SmartConnect
        import pyotp
    except ImportError as error:
        raise SystemExit("Angel One Python dependencies are not installed") from error
    # smartapi-python logs complete request headers on some provider errors.
    # Disable third-party logging in this one-shot process before authentication
    # so backend-only credentials cannot be copied into terminals or CI logs.
    logging.disable(logging.CRITICAL)
    request = urllib.request.Request(
        args.instrument_master_url,
        headers={"User-Agent": "stock-intelligence-private/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        master = json.loads(response.read())
    future, option = resolve_nifty_contracts(master, as_of=date.today())
    client = SmartConnect(settings.angelone_api_key or "")
    session = client.generateSession(
        settings.angelone_client_code or "",
        settings.angelone_pin or "",
        pyotp.TOTP(settings.angelone_totp_secret or "").now(),
    )
    if not isinstance(session, dict) or session.get("status") is not True:
        raise SystemExit("Angel One authentication failed safely")
    option_expiry = datetime.strptime(str(option["expiry"]).upper(), "%d%b%Y").strftime("%d%b%Y").upper()
    snapshot = build_derivatives_snapshot(
        observed_at=datetime.now(timezone.utc),
        future_contract=future,
        option_contract=option,
        market_response=client.getMarketData(
            "FULL",
            {
                "NFO": [str(future["token"])],
                "NSE": [settings.angelone_nifty_websocket_token],
            },
        ),
        pcr_response=client.putCallRatio(),
        greeks_response=client.optionGreek({"name": "NIFTY", "expirydate": option_expiry}),
        spot_token=settings.angelone_nifty_websocket_token,
    )
    inserted = SQLiteDerivativesLedger(args.database).append(snapshot)
    output = snapshot.to_contract()
    output["inserted"] = inserted
    output["database"] = str(args.database)
    output["automatic_trading_enabled"] = False
    output["official_signal_available"] = False
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

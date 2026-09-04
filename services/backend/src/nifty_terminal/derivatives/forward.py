"""Fail-closed point-in-time NIFTY derivatives context collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DerivativesSnapshot:
    snapshot_id: str
    observed_at: str
    provider: str
    spot_symbol: str
    spot_token: str
    spot_ltp: float | None
    futures_symbol: str
    futures_token: str
    futures_expiry: str
    futures_ltp: float
    futures_volume: float
    futures_open_interest: float
    futures_buy_quantity: float
    futures_sell_quantity: float
    futures_book_imbalance: float
    futures_basis_points: float | None
    futures_basis_bps: float | None
    provider_put_call_ratio: float | None
    option_expiry: str
    atm_implied_volatility: float | None
    delta25_put_call_iv_skew: float | None
    option_volume_put_call_ratio: float | None
    unavailable_sources: tuple[str, ...]
    research_only: bool = True

    def to_contract(self) -> dict[str, object]:
        return asdict(self)


def resolve_nifty_contracts(master: object, *, as_of: date) -> tuple[dict, dict]:
    if not isinstance(master, list):
        raise ValueError("Angel One instrument master must be a list")
    futures = []
    options = []
    for raw in master:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("exch_seg", "")).upper() != "NFO":
            continue
        if str(raw.get("name", "")).upper() != "NIFTY":
            continue
        try:
            expiry = datetime.strptime(str(raw.get("expiry", "")).upper(), "%d%b%Y").date()
        except ValueError:
            continue
        if expiry < as_of:
            continue
        item = (expiry, str(raw.get("symbol", "")), dict(raw))
        kind = str(raw.get("instrumenttype", "")).upper()
        if kind == "FUTIDX":
            futures.append(item)
        elif kind == "OPTIDX":
            options.append(item)
    if not futures or not options:
        raise ValueError("Current NIFTY future and option contracts were not found")
    return min(futures, key=lambda item: (item[0], item[1]))[2], min(
        options, key=lambda item: (item[0], item[1])
    )[2]


def build_derivatives_snapshot(
    *,
    observed_at: datetime,
    future_contract: Mapping[str, object],
    option_contract: Mapping[str, object],
    market_response: object,
    pcr_response: object,
    greeks_response: object,
    spot_token: str | None = None,
    spot_symbol: str = "NIFTY 50",
) -> DerivativesSnapshot:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    quote_rows = _response_rows(market_response, nested_key="fetched")
    token = str(future_contract.get("token", ""))
    quote = next((row for row in quote_rows if str(row.get("symbolToken", row.get("symboltoken", ""))) == token), None)
    if quote is None:
        raise ValueError("Angel One full quote omitted the selected NIFTY future")
    ltp = _number(quote, "ltp")
    volume = _number(quote, "tradeVolume")
    oi = _number(quote, "opnInterest")
    buy = _number(quote, "totBuyQuan")
    sell = _number(quote, "totSellQuan")
    if min(ltp, volume, oi, buy, sell) < 0 or ltp <= 0:
        raise ValueError("NIFTY futures quote contains invalid values")
    imbalance = (buy - sell) / (buy + sell) if buy + sell else 0.0
    unavailable = []
    spot_quote = next(
        (
            row
            for row in quote_rows
            if spot_token is not None
            and str(row.get("symbolToken", row.get("symboltoken", "")))
            == spot_token
        ),
        None,
    )
    spot_ltp = _optional_number(spot_quote, "ltp")
    if spot_ltp is None or spot_ltp <= 0:
        spot_ltp = None
        unavailable.append("SPOT_QUOTE")
    basis_points = ltp - spot_ltp if spot_ltp is not None else None
    basis_bps = basis_points / spot_ltp * 10_000 if spot_ltp is not None else None
    pcr_rows = _optional_response_rows(pcr_response)
    if not pcr_rows:
        unavailable.append("PUT_CALL_RATIO")
    future_symbol = str(future_contract.get("symbol", ""))
    pcr_row = next((row for row in pcr_rows if str(row.get("tradingSymbol", "")) == future_symbol), None)
    if pcr_row is None:
        pcr_row = next((row for row in pcr_rows if str(row.get("tradingSymbol", "")).upper().startswith("NIFTY")), None)
    provider_pcr = _optional_number(pcr_row, "pcr") if pcr_row else None
    greek_rows = _optional_response_rows(greeks_response)
    if not greek_rows:
        unavailable.append("OPTION_GREEKS")
    calls = [row for row in greek_rows if str(row.get("optionType", "")).upper() == "CE"]
    puts = [row for row in greek_rows if str(row.get("optionType", "")).upper() == "PE"]
    atm_call = _closest_delta(calls, 0.5)
    atm_put = _closest_delta(puts, -0.5)
    put25 = _closest_delta(puts, -0.25)
    call25 = _closest_delta(calls, 0.25)
    atm_iv = _mean_optional((_optional_number(atm_call, "impliedVolatility"), _optional_number(atm_put, "impliedVolatility")))
    put25_iv = _optional_number(put25, "impliedVolatility")
    call25_iv = _optional_number(call25, "impliedVolatility")
    skew = put25_iv - call25_iv if put25_iv is not None and call25_iv is not None else None
    call_volume = sum(_optional_number(row, "tradeVolume") or 0.0 for row in calls)
    put_volume = sum(_optional_number(row, "tradeVolume") or 0.0 for row in puts)
    option_volume_pcr = put_volume / call_volume if call_volume > 0 else None
    values = {
        "observed_at": observed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provider": "angelone",
        "spot_symbol": spot_symbol,
        "spot_token": spot_token or "",
        "spot_ltp": spot_ltp,
        "futures_symbol": future_symbol,
        "futures_token": token,
        "futures_expiry": str(future_contract.get("expiry", "")),
        "futures_ltp": ltp,
        "futures_volume": volume,
        "futures_open_interest": oi,
        "futures_buy_quantity": buy,
        "futures_sell_quantity": sell,
        "futures_book_imbalance": imbalance,
        "futures_basis_points": basis_points,
        "futures_basis_bps": basis_bps,
        "provider_put_call_ratio": provider_pcr,
        "option_expiry": str(option_contract.get("expiry", "")),
        "atm_implied_volatility": atm_iv,
        "delta25_put_call_iv_skew": skew,
        "option_volume_put_call_ratio": option_volume_pcr,
        "unavailable_sources": tuple(unavailable),
        "research_only": True,
    }
    identity = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return DerivativesSnapshot(
        snapshot_id=hashlib.sha256(identity.encode()).hexdigest(),
        **values,
    )


class SQLiteDerivativesLedger:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, snapshot: DerivativesSnapshot) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS derivatives_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    observed_at TEXT NOT NULL,
                    futures_symbol TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO derivatives_snapshots VALUES (?, ?, ?, ?)",
                (
                    snapshot.snapshot_id,
                    snapshot.observed_at,
                    snapshot.futures_symbol,
                    json.dumps(snapshot.to_contract(), sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def load_contracts(self) -> tuple[dict[str, object], ...]:
        if not self._path.exists():
            return ()
        connection = sqlite3.connect(self._path)
        try:
            rows = connection.execute(
                "SELECT payload_json FROM derivatives_snapshots ORDER BY observed_at, snapshot_id"
            ).fetchall()
        finally:
            connection.close()
        return tuple(json.loads(str(row[0])) for row in rows)


def _response_rows(response: object, *, nested_key: str | None = None) -> tuple[Mapping, ...]:
    if not isinstance(response, Mapping) or response.get("status") is not True:
        code = response.get("errorcode", "INVALID") if isinstance(response, Mapping) else "INVALID"
        raise ValueError(f"Angel One research-data response failed safely [{code}]")
    data = response.get("data")
    if nested_key is not None and isinstance(data, Mapping):
        data = data.get(nested_key)
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError("Angel One research-data response omitted rows")
    return tuple(row for row in data if isinstance(row, Mapping))


def _optional_response_rows(response: object) -> tuple[Mapping, ...]:
    if not isinstance(response, Mapping) or response.get("status") is not True:
        return ()
    data = response.get("data")
    if data is None:
        return ()
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError("Angel One optional research-data response is malformed")
    return tuple(row for row in data if isinstance(row, Mapping))


def _number(row: Mapping[str, object], key: str) -> float:
    value = _optional_number(row, key)
    if value is None:
        raise ValueError(f"Angel One derivatives field missing: {key}")
    return value


def _optional_number(row: Mapping[str, object] | None, key: str) -> float | None:
    if row is None or row.get(key) in (None, ""):
        return None
    value = float(row[key])
    if not np_isfinite(value):
        raise ValueError(f"Angel One derivatives field is not finite: {key}")
    return value


def _closest_delta(rows: list[Mapping], target: float) -> Mapping | None:
    valid = [(abs(value - target), row) for row in rows if (value := _optional_number(row, "delta")) is not None]
    return min(valid, key=lambda item: item[0])[1] if valid else None


def _mean_optional(values: tuple[float | None, ...]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))

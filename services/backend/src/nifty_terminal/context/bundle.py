"""Immutable, content-addressed context-history bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ContextBar:
    opens_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None

    def to_contract(self) -> dict[str, object]:
        return {
            "opens_at": self.opens_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "open": format(self.open, "f"),
            "high": format(self.high, "f"),
            "low": format(self.low, "f"),
            "close": format(self.close, "f"),
            "volume": format(self.volume, "f") if self.volume is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ContextInstrument:
    instrument_id: str
    provider: str
    exchange: str
    token: str
    asset_kind: str
    bars: tuple[ContextBar, ...]
    expected_minutes: int
    excluded_out_of_session: int

    @property
    def coverage_ratio(self) -> float:
        return len(self.bars) / self.expected_minutes if self.expected_minutes else 0.0

    def to_contract(self) -> dict[str, object]:
        return {
            "instrument_id": self.instrument_id,
            "provider": self.provider,
            "exchange": self.exchange,
            "token": self.token,
            "asset_kind": self.asset_kind,
            "expected_minutes": self.expected_minutes,
            "observed_minutes": len(self.bars),
            "coverage_ratio": self.coverage_ratio,
            "excluded_out_of_session": self.excluded_out_of_session,
            "bars": [item.to_contract() for item in self.bars],
        }


@dataclass(frozen=True, slots=True)
class ContextBundle:
    schema_version: int
    provider: str
    requested_from: str
    requested_through: str
    acquired_at: str
    instruments: tuple[ContextInstrument, ...]
    source_notes: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "requested_from": self.requested_from,
            "requested_through": self.requested_through,
            "acquired_at": self.acquired_at,
            "instruments": [item.to_contract() for item in self.instruments],
            "source_notes": list(self.source_notes),
        }


def bundle_sha256(bundle: ContextBundle) -> str:
    payload = bundle.to_contract().copy()
    payload.pop("acquired_at", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_bundle(path: Path, bundle: ContextBundle) -> str:
    digest = bundle_sha256(bundle)
    payload = bundle.to_contract()
    payload["content_sha256"] = digest
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_bundle(path)
        if bundle_sha256(existing) != digest:
            raise FileExistsError(f"Context bundle already exists with different content: {path}")
        return digest
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as file:
            file.write(encoded)
    return digest


def read_bundle(path: Path) -> ContextBundle:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    instruments = []
    for item in payload["instruments"]:
        bars = tuple(
            ContextBar(
                opens_at=_time(row["opens_at"]),
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=Decimal(row["volume"]) if row.get("volume") is not None else None,
            )
            for row in item["bars"]
        )
        instruments.append(
            ContextInstrument(
                instrument_id=item["instrument_id"],
                provider=item["provider"],
                exchange=item["exchange"],
                token=str(item["token"]),
                asset_kind=item["asset_kind"],
                bars=bars,
                expected_minutes=int(item["expected_minutes"]),
                excluded_out_of_session=int(item["excluded_out_of_session"]),
            )
        )
    bundle = ContextBundle(
        schema_version=int(payload["schema_version"]),
        provider=payload["provider"],
        requested_from=payload["requested_from"],
        requested_through=payload["requested_through"],
        acquired_at=payload["acquired_at"],
        instruments=tuple(instruments),
        source_notes=tuple(payload.get("source_notes", ())),
    )
    expected = payload.get("content_sha256")
    if expected and bundle_sha256(bundle) != expected:
        raise ValueError("Context bundle SHA-256 verification failed")
    return bundle


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Context timestamp must contain an offset")
    return parsed.astimezone(timezone.utc)

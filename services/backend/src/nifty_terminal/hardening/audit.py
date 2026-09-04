"""Append-only, hash-chained local security and release audit ledger."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from collections.abc import Iterator


GENESIS_HASH = "0" * 64


class SQLiteAuditLedger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def append(
        self,
        *,
        event_id: str,
        occurred_at: datetime,
        category: str,
        action: str,
        actor: str,
        details: dict[str, object],
    ) -> bool:
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if not all(value.strip() for value in (event_id, category, action, actor)):
            raise ValueError("Audit identity fields are required")
        payload = json.dumps(details, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT event_hash, payload_json FROM security_audit_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                expected = self._hash_for_existing(connection, event_id, occurred_at, category, action, actor, payload)
                if existing["event_hash"] != expected or existing["payload_json"] != payload:
                    raise ValueError("immutable audit event identity already contains different data")
                return False
            previous = connection.execute(
                "SELECT event_hash FROM security_audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous["event_hash"] if previous else GENESIS_HASH
            event_hash = _event_hash(previous_hash, event_id, occurred_at, category, action, actor, payload)
            connection.execute(
                """
                INSERT INTO security_audit_events
                    (event_id, occurred_at, category, action, actor, payload_json, previous_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    occurred_at.isoformat(),
                    category,
                    action,
                    actor,
                    payload,
                    previous_hash,
                    event_hash,
                ),
            )
        return True

    def verify_chain(self) -> bool:
        previous_hash = GENESIS_HASH
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, occurred_at, category, action, actor, payload_json,
                       previous_hash, event_hash
                FROM security_audit_events ORDER BY sequence ASC
                """
            ).fetchall()
        for row in rows:
            occurred_at = datetime.fromisoformat(row["occurred_at"])
            expected = _event_hash(
                previous_hash,
                row["event_id"],
                occurred_at,
                row["category"],
                row["action"],
                row["actor"],
                row["payload_json"],
            )
            if row["previous_hash"] != previous_hash or row["event_hash"] != expected:
                return False
            previous_hash = row["event_hash"]
        return True

    def _hash_for_existing(
        self,
        connection: sqlite3.Connection,
        event_id: str,
        occurred_at: datetime,
        category: str,
        action: str,
        actor: str,
        payload: str,
    ) -> str:
        row = connection.execute(
            "SELECT previous_hash FROM security_audit_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return _event_hash(row["previous_hash"], event_id, occurred_at, category, action, actor, payload)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    occurred_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT UNIQUE NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS security_audit_events_no_update
                BEFORE UPDATE ON security_audit_events
                BEGIN SELECT RAISE(ABORT, 'append-only audit ledger'); END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS security_audit_events_no_delete
                BEFORE DELETE ON security_audit_events
                BEGIN SELECT RAISE(ABORT, 'append-only audit ledger'); END
                """
            )


def _event_hash(
    previous_hash: str,
    event_id: str,
    occurred_at: datetime,
    category: str,
    action: str,
    actor: str,
    payload: str,
) -> str:
    canonical = "|".join(
        (previous_hash, event_id, occurred_at.isoformat(), category, action, actor, payload)
    )
    return sha256(canonical.encode("utf-8")).hexdigest()

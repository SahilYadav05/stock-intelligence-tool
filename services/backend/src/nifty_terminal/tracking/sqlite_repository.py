"""Append-only local SQLite ledger for Step 9 tracking records."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from collections.abc import Iterator

from nifty_terminal.tracking.models import (
    PaperTrade,
    PaperTradeEvent,
    PredictionAssessment,
    TrackedPrediction,
)


class SQLiteTrackingLedger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def append_prediction(self, item: TrackedPrediction) -> bool:
        return self._append(
            "tracked_predictions",
            "prediction_id",
            item.prediction_id,
            item.instrument_id,
            item.registered_at.isoformat(),
            item.to_contract(),
        )

    def append_assessment(self, item: PredictionAssessment) -> bool:
        return self._append(
            "prediction_assessments",
            "assessment_id",
            item.assessment_id,
            item.instrument_id,
            item.assessed_at.isoformat(),
            item.to_contract(),
        )

    def append_paper_trade(self, item: PaperTrade) -> bool:
        return self._append(
            "paper_trades",
            "paper_trade_id",
            item.paper_trade_id,
            item.instrument_id,
            item.created_at.isoformat(),
            item.to_contract(),
        )

    def append_paper_event(self, item: PaperTradeEvent, *, instrument_id: str) -> bool:
        return self._append(
            "paper_trade_events",
            "event_id",
            item.event_id,
            instrument_id,
            item.occurred_at.isoformat(),
            item.to_contract(),
        )

    def load_contracts(self, table: str, *, instrument_id: str) -> tuple[dict[str, object], ...]:
        if table not in {
            "tracked_predictions",
            "prediction_assessments",
            "paper_trades",
            "paper_trade_events",
        }:
            raise ValueError("unsupported tracking table")
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload_json FROM {table} WHERE instrument_id = ? ORDER BY occurred_at ASC",
                (instrument_id,),
            ).fetchall()
        return tuple(json.loads(row["payload_json"]) for row in rows)

    def _append(
        self,
        table: str,
        identity_column: str,
        identity: str,
        instrument_id: str,
        occurred_at: str,
        payload: dict[str, object],
    ) -> bool:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            existing = connection.execute(
                f"SELECT payload_json FROM {table} WHERE {identity_column} = ?",
                (identity,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != serialized:
                    raise ValueError("immutable tracking identity already contains different data")
                return False
            connection.execute(
                f"INSERT INTO {table} ({identity_column}, instrument_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
                (identity, instrument_id, occurred_at, serialized),
            )
        return True

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
            for table, identity in (
                ("tracked_predictions", "prediction_id"),
                ("prediction_assessments", "assessment_id"),
                ("paper_trades", "paper_trade_id"),
                ("paper_trade_events", "event_id"),
            ):
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        {identity} TEXT PRIMARY KEY,
                        instrument_id TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_instrument_time_idx ON {table} (instrument_id, occurred_at)"
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN SELECT RAISE(ABORT, 'append-only table'); END
                    """
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN SELECT RAISE(ABORT, 'append-only table'); END
                    """
                )

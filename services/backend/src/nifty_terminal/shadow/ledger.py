"""Append-only SQLite ledger for shadow predictions and later assessments."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from collections.abc import Iterator


class SQLiteShadowLedger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def append_prediction(self, payload: dict[str, object]) -> bool:
        serialized = _canonical(payload)
        prediction_id = str(payload["prediction_id"])
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM shadow_predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != serialized:
                    raise ValueError("Immutable shadow prediction identity collision")
                return False
            connection.execute(
                """
                INSERT INTO shadow_predictions
                    (prediction_id, snapshot_id, decision_time, due_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    str(payload["snapshot_id"]),
                    str(payload["decision_time"]),
                    str(payload["outcome_due_at"]),
                    serialized,
                ),
            )
        return True

    def append_assessment(self, payload: dict[str, object]) -> bool:
        serialized = _canonical(payload)
        prediction_id = str(payload["prediction_id"])
        assessment_id = str(payload["assessment_id"])
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM shadow_assessments WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != serialized:
                    raise ValueError("Immutable shadow assessment identity collision")
                return False
            connection.execute(
                """
                INSERT INTO shadow_assessments
                    (assessment_id, prediction_id, assessed_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (assessment_id, prediction_id, str(payload["assessed_at"]), serialized),
            )
        return True

    def pending(self, *, due_at_or_before: datetime) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.payload_json
                FROM shadow_predictions p
                LEFT JOIN shadow_assessments a ON a.prediction_id = p.prediction_id
                WHERE a.prediction_id IS NULL AND p.due_at <= ?
                ORDER BY p.decision_time ASC
                """,
                (due_at_or_before.isoformat(),),
            ).fetchall()
        return tuple(json.loads(row["payload_json"]) for row in rows)

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM shadow_predictions) AS predictions,
                    (SELECT COUNT(*) FROM shadow_assessments) AS assessments,
                    (SELECT MAX(decision_time) FROM shadow_predictions) AS latest_decision
                """
            ).fetchone()
        return {
            "prediction_count": int(row["predictions"]),
            "assessment_count": int(row["assessments"]),
            "pending_assessment_count": int(row["predictions"]) - int(row["assessments"]),
            "latest_prediction_decision_time": row["latest_decision"],
        }

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_predictions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    decision_time TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shadow_assessments (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    assessment_id TEXT UNIQUE NOT NULL,
                    prediction_id TEXT UNIQUE NOT NULL,
                    assessed_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES shadow_predictions(prediction_id)
                );
                CREATE TRIGGER IF NOT EXISTS shadow_predictions_no_update
                BEFORE UPDATE ON shadow_predictions
                BEGIN SELECT RAISE(ABORT, 'append-only shadow predictions'); END;
                CREATE TRIGGER IF NOT EXISTS shadow_predictions_no_delete
                BEFORE DELETE ON shadow_predictions
                BEGIN SELECT RAISE(ABORT, 'append-only shadow predictions'); END;
                CREATE TRIGGER IF NOT EXISTS shadow_assessments_no_update
                BEFORE UPDATE ON shadow_assessments
                BEGIN SELECT RAISE(ABORT, 'append-only shadow assessments'); END;
                CREATE TRIGGER IF NOT EXISTS shadow_assessments_no_delete
                BEFORE DELETE ON shadow_assessments
                BEGIN SELECT RAISE(ABORT, 'append-only shadow assessments'); END;
                """
            )


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

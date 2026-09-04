"""In-memory latest analysis projection keyed by immutable snapshot identity."""

from __future__ import annotations

from threading import RLock

from nifty_terminal.dashboard.models import AnalysisView


class InMemoryAnalysisReadModel:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_snapshot: dict[str, AnalysisView] = {}

    def put(self, view: AnalysisView) -> None:
        with self._lock:
            existing = self._by_snapshot.get(view.snapshot_id)
            if existing is not None and existing.generated_at > view.generated_at:
                raise ValueError("Cannot replace analysis with an older projection")
            self._by_snapshot[view.snapshot_id] = view

    def get(self, snapshot_id: str) -> AnalysisView | None:
        with self._lock:
            return self._by_snapshot.get(snapshot_id)

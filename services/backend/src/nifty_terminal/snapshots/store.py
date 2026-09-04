"""Append-only storage boundary for immutable market-state snapshots."""

from nifty_terminal.snapshots.models import MarketStateSnapshot


class InMemorySnapshotStore:
    def __init__(self) -> None:
        self._snapshots: list[MarketStateSnapshot] = []
        self._ids: set[str] = set()

    def append(self, snapshot: MarketStateSnapshot) -> None:
        if snapshot.snapshot_id in self._ids:
            raise ValueError(f"Snapshot already stored: {snapshot.snapshot_id}")
        self._snapshots.append(snapshot)
        self._ids.add(snapshot.snapshot_id)

    def all(self) -> tuple[MarketStateSnapshot, ...]:
        return tuple(self._snapshots)

    def latest(self, instrument_id: str) -> MarketStateSnapshot | None:
        matches = [item for item in self._snapshots if item.instrument_id == instrument_id]
        return matches[-1] if matches else None

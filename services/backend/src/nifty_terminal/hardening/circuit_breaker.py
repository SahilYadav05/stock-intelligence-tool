"""Explicit operator and automated circuit-breaker state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock


@dataclass(frozen=True, slots=True)
class CircuitBreakerState:
    open: bool
    reason: str | None
    changed_at: datetime | None


class SignalCircuitBreaker:
    def __init__(self, *, initially_open: bool = True, reason: str = "STARTUP_FAIL_CLOSED") -> None:
        self._lock = RLock()
        self._state = CircuitBreakerState(initially_open, reason if initially_open else None, None)

    def trip(self, *, reason: str, changed_at: datetime) -> CircuitBreakerState:
        if not reason:
            raise ValueError("Circuit-breaker reason is required")
        with self._lock:
            self._state = CircuitBreakerState(True, reason, changed_at)
            return self._state

    def reset(self, *, changed_at: datetime, release_gate_passed: bool) -> CircuitBreakerState:
        if not release_gate_passed:
            raise ValueError("Circuit breaker cannot reset before release gates pass")
        with self._lock:
            self._state = CircuitBreakerState(False, None, changed_at)
            return self._state

    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state

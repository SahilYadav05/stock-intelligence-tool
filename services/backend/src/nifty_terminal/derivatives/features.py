"""Causal feature materialization and readiness gates for derivatives snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time
import hashlib
import json
import math
from typing import Mapping, Sequence

from nifty_terminal.calendar.nse import IST

DERIVATIVES_FEATURE_VERSION = "derivatives_context.v1"
DERIVATIVES_FEATURE_NAMES = (
    "derivatives__spot_return_1",
    "derivatives__futures_return_1",
    "derivatives__basis_bps",
    "derivatives__basis_change_bps",
    "derivatives__volume_delta_log1p",
    "derivatives__open_interest_change_pct",
    "derivatives__book_imbalance",
    "derivatives__book_imbalance_change",
    "derivatives__provider_pcr",
    "derivatives__provider_pcr_change",
    "derivatives__atm_iv",
    "derivatives__atm_iv_change",
    "derivatives__delta25_iv_skew",
    "derivatives__delta25_iv_skew_change",
    "derivatives__option_volume_pcr",
)
DERIVATIVES_FEATURE_SET_HASH = hashlib.sha256(
    json.dumps(
        {
            "version": DERIVATIVES_FEATURE_VERSION,
            "features": DERIVATIVES_FEATURE_NAMES,
            "point_in_time": True,
            "previous_snapshot_max_gap_minutes": 15,
            "regular_session_ist": "09:15-15:30",
            "contract_roll_resets_deltas": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class DerivativesFeatureRow:
    snapshot_id: str
    observed_at: str
    session_date: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float | None, ...]
    complete_core: bool
    options_complete: bool
    eligible_regular_session: bool

    def to_contract(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DerivativesReadinessReport:
    snapshot_count: int
    regular_session_snapshot_count: int
    feature_row_count: int
    complete_core_row_count: int
    represented_sessions: int
    calendar_span_days: int
    options_complete_share: float
    minimum_complete_rows: int
    minimum_sessions: int
    minimum_calendar_span_days: int
    minimum_options_complete_share: float
    blockers: tuple[str, ...]

    @property
    def ready_for_model_research(self) -> bool:
        return not self.blockers

    def to_contract(self) -> dict[str, object]:
        return asdict(self) | {
            "ready_for_model_research": self.ready_for_model_research,
            "research_only": True,
            "official_signal_available": False,
        }


def build_derivatives_feature_rows(
    snapshots: Sequence[Mapping[str, object]],
) -> tuple[DerivativesFeatureRow, ...]:
    ordered = sorted(snapshots, key=lambda item: (_observed_at(item), str(item.get("snapshot_id", ""))))
    rows = []
    previous: Mapping[str, object] | None = None
    previous_at: datetime | None = None
    for snapshot in ordered:
        observed_at = _observed_at(snapshot)
        local = observed_at.astimezone(IST)
        regular = local.weekday() < 5 and time(9, 15) <= local.time() <= time(15, 30)
        comparable = (
            previous is not None
            and previous_at is not None
            and previous_at.astimezone(IST).date() == local.date()
            and snapshot.get("futures_symbol") == previous.get("futures_symbol")
            and 0 < (observed_at - previous_at).total_seconds() <= 15 * 60
        )
        values = _feature_values(snapshot, previous if comparable else None)
        core_indices = tuple(range(0, 8))
        option_indices = tuple(range(8, len(values)))
        rows.append(
            DerivativesFeatureRow(
                snapshot_id=str(snapshot.get("snapshot_id", "")),
                observed_at=observed_at.isoformat().replace("+00:00", "Z"),
                session_date=str(local.date()),
                feature_names=DERIVATIVES_FEATURE_NAMES,
                feature_values=values,
                complete_core=regular
                and comparable
                and all(values[index] is not None for index in core_indices),
                options_complete=regular
                and all(values[index] is not None for index in option_indices),
                eligible_regular_session=regular,
            )
        )
        previous = snapshot
        previous_at = observed_at
    return tuple(rows)


def evaluate_derivatives_readiness(
    snapshots: Sequence[Mapping[str, object]],
    *,
    minimum_complete_rows: int = 3_000,
    minimum_sessions: int = 60,
    minimum_calendar_span_days: int = 90,
    minimum_options_complete_share: float = 0.80,
) -> DerivativesReadinessReport:
    rows = build_derivatives_feature_rows(snapshots)
    regular = [row for row in rows if row.eligible_regular_session]
    complete = [row for row in rows if row.complete_core]
    sessions = sorted({row.session_date for row in complete})
    span = (
        (
            datetime.fromisoformat(sessions[-1]).date()
            - datetime.fromisoformat(sessions[0]).date()
        ).days
        + 1
        if sessions
        else 0
    )
    option_share = (
        sum(row.options_complete for row in regular) / len(regular) if regular else 0.0
    )
    blockers = []
    if len(complete) < minimum_complete_rows:
        blockers.append("DERIVATIVES_COMPLETE_ROW_SUPPORT_TOO_LOW")
    if len(sessions) < minimum_sessions:
        blockers.append("DERIVATIVES_SESSION_SUPPORT_TOO_LOW")
    if span < minimum_calendar_span_days:
        blockers.append("DERIVATIVES_CALENDAR_SPAN_TOO_SHORT")
    if option_share < minimum_options_complete_share:
        blockers.append("OPTIONS_CONTEXT_COMPLETENESS_TOO_LOW")
    return DerivativesReadinessReport(
        snapshot_count=len(snapshots),
        regular_session_snapshot_count=len(regular),
        feature_row_count=len(rows),
        complete_core_row_count=len(complete),
        represented_sessions=len(sessions),
        calendar_span_days=span,
        options_complete_share=option_share,
        minimum_complete_rows=minimum_complete_rows,
        minimum_sessions=minimum_sessions,
        minimum_calendar_span_days=minimum_calendar_span_days,
        minimum_options_complete_share=minimum_options_complete_share,
        blockers=tuple(blockers),
    )


def _feature_values(
    current: Mapping[str, object], previous: Mapping[str, object] | None
) -> tuple[float | None, ...]:
    spot = _number(current, "spot_ltp")
    future = _number(current, "futures_ltp")
    basis = _number(current, "futures_basis_bps")
    imbalance = _number(current, "futures_book_imbalance")
    pcr = _number(current, "provider_put_call_ratio")
    atm_iv = _number(current, "atm_implied_volatility")
    skew = _number(current, "delta25_put_call_iv_skew")
    option_volume_pcr = _number(current, "option_volume_put_call_ratio")
    if previous is None:
        spot_return = future_return = basis_change = None
        volume_delta = oi_change = imbalance_change = None
        pcr_change = iv_change = skew_change = None
    else:
        spot_return = _return(spot, _number(previous, "spot_ltp"))
        future_return = _return(future, _number(previous, "futures_ltp"))
        basis_change = _difference(basis, _number(previous, "futures_basis_bps"))
        current_volume = _number(current, "futures_volume")
        previous_volume = _number(previous, "futures_volume")
        volume_delta = (
            math.log1p(max(current_volume - previous_volume, 0.0))
            if current_volume is not None and previous_volume is not None
            else None
        )
        oi_change = _return(
            _number(current, "futures_open_interest"),
            _number(previous, "futures_open_interest"),
        )
        imbalance_change = _difference(
            imbalance, _number(previous, "futures_book_imbalance")
        )
        pcr_change = _difference(pcr, _number(previous, "provider_put_call_ratio"))
        iv_change = _difference(atm_iv, _number(previous, "atm_implied_volatility"))
        skew_change = _difference(
            skew, _number(previous, "delta25_put_call_iv_skew")
        )
    return (
        spot_return,
        future_return,
        basis,
        basis_change,
        volume_delta,
        oi_change,
        imbalance,
        imbalance_change,
        pcr,
        pcr_change,
        atm_iv,
        iv_change,
        skew,
        skew_change,
        option_volume_pcr,
    )


def _observed_at(snapshot: Mapping[str, object]) -> datetime:
    raw = str(snapshot.get("observed_at", ""))
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Derivatives snapshot has invalid observed_at") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Derivatives snapshot observed_at must be timezone-aware")
    return value


def _number(row: Mapping[str, object], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _return(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None
    return current / previous - 1.0


def _difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous

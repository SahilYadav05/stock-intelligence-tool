"""Point-in-time-safe quality checks for historical minute bars."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.history.models import (
    HistoricalBatch,
    HistoricalQualityReport,
    QualityStatus,
)


class HistoricalQualityInspector:
    def __init__(self, calendar: NseSessionCalendar) -> None:
        self._calendar = calendar

    def inspect(self, batch: HistoricalBatch) -> HistoricalQualityReport:
        rows = batch.rows
        errors: list[str] = []
        warnings: list[str] = []
        provider_ids = Counter(item.provider_bar_id for item in rows)
        duplicate_provider_ids = sum(count - 1 for count in provider_ids.values() if count > 1)
        if duplicate_provider_ids:
            errors.append("DUPLICATE_PROVIDER_BAR_IDS")

        first_seen_opens: list[object] = []
        seen_opens: set[object] = set()
        for item in rows:
            if item.opens_at not in seen_opens:
                first_seen_opens.append(item.opens_at)
                seen_opens.add(item.opens_at)
        out_of_order = sum(
            current < previous
            for previous, current in zip(first_seen_opens, first_seen_opens[1:], strict=False)
        )
        if out_of_order:
            errors.append("OUT_OF_ORDER_ROWS")

        out_of_request = sum(
            not (batch.request.starts_at <= item.opens_at < batch.request.ends_at)
            for item in rows
        )
        if out_of_request:
            errors.append("ROWS_OUTSIDE_REQUEST_INTERVAL")

        revisions: dict[object, list[int]] = defaultdict(list)
        for item in rows:
            revisions[item.opens_at].append(item.provider_revision)
            try:
                bucket = self._calendar.bucket_for(item.opens_at, Timeframe.M1)
                if bucket.opens_at != item.opens_at or bucket.closes_at != item.closes_at:
                    errors.append("MISALIGNED_SESSION_BAR")
            except ValueError:
                errors.append("OUT_OF_SESSION_BAR")
        for values in revisions.values():
            if values != sorted(set(values)) or values[0] < 1:
                errors.append("INVALID_CORRECTION_SEQUENCE")
                break

        unique_opens = sorted(revisions)
        expected_opens = self._expected_opens(batch)
        missing_minutes = len(expected_opens.difference(unique_opens))
        if missing_minutes:
            warnings.append("INTRADAY_MINUTE_GAPS")
        if not rows:
            errors.append("EMPTY_DATASET")

        errors = sorted(set(errors))
        status = QualityStatus.REJECTED if errors else (
            QualityStatus.DEGRADED if warnings else QualityStatus.PASS
        )
        return HistoricalQualityReport(
            status=status,
            total_rows=len(rows),
            unique_minute_buckets=len(unique_opens),
            correction_rows=len(rows) - len(unique_opens),
            missing_minutes=missing_minutes,
            duplicate_provider_ids=duplicate_provider_ids,
            out_of_order_rows=out_of_order,
            out_of_request_rows=out_of_request,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _expected_opens(self, batch: HistoricalBatch) -> set[object]:
        expected: set[object] = set()
        local_start = batch.request.starts_at.astimezone(IST).date()
        local_end = batch.request.ends_at.astimezone(IST).date()
        current_date = local_start
        while current_date <= local_end:
            session = self._calendar.session_for_date(current_date)
            if session is not None:
                cursor = session.opens_at
                while cursor < session.closes_at:
                    utc_cursor = cursor.astimezone(batch.request.starts_at.tzinfo)
                    if batch.request.starts_at <= utc_cursor < batch.request.ends_at:
                        expected.add(utc_cursor)
                    cursor += timedelta(minutes=1)
            current_date += timedelta(days=1)
        return expected

from __future__ import annotations

from datetime import date, datetime, time, timezone
from unittest import TestCase

from nifty_terminal.calendar.nse import IST, NseSessionCalendar, SessionKind
from nifty_terminal.domain.candle import Timeframe


class NseSessionCalendarTests(TestCase):
    def test_buckets_are_anchored_to_0915_ist(self) -> None:
        calendar = NseSessionCalendar()
        instant = datetime(2026, 8, 24, 9, 17, tzinfo=IST)

        five = calendar.bucket_for(instant, Timeframe.M5)
        hourly = calendar.bucket_for(instant, Timeframe.H1)

        self.assertEqual(five.opens_at, datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc))
        self.assertEqual(five.closes_at, datetime(2026, 8, 24, 3, 50, tzinfo=timezone.utc))
        self.assertEqual(hourly.opens_at, datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc))
        self.assertEqual(hourly.closes_at, datetime(2026, 8, 24, 4, 45, tzinfo=timezone.utc))

    def test_trailing_hour_bucket_is_explicitly_partial(self) -> None:
        calendar = NseSessionCalendar()
        bucket = calendar.bucket_for(
            datetime(2026, 7, 31, 15, 20, tzinfo=IST), Timeframe.H1
        )

        self.assertTrue(bucket.is_partial)
        self.assertEqual(bucket.expected_minutes, 15)

    def test_cas_day_ends_on_a_complete_hour_bucket(self) -> None:
        calendar = NseSessionCalendar()
        bucket = calendar.bucket_for(
            datetime(2026, 8, 24, 15, 14, tzinfo=IST), Timeframe.H1
        )

        self.assertFalse(bucket.is_partial)
        self.assertEqual(bucket.expected_minutes, 60)
        self.assertEqual(bucket.closes_at.astimezone(IST), datetime(2026, 8, 24, 15, 15, tzinfo=IST))

    def test_weekends_holidays_and_special_sessions_are_explicit(self) -> None:
        holiday = date(2026, 8, 25)
        special_date = date(2026, 8, 30)
        calendar = NseSessionCalendar(
            holidays=frozenset({holiday}),
            special_sessions={special_date: (time(18, 0), time(19, 0))},
        )

        self.assertIsNone(calendar.session_for_date(date(2026, 8, 29)))
        self.assertIsNone(calendar.session_for_date(holiday))
        special = calendar.session_for_date(special_date)
        self.assertIsNotNone(special)
        self.assertEqual(special.kind, SessionKind.SPECIAL)  # type: ignore[union-attr]

    def test_outside_session_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            NseSessionCalendar().bucket_for(
                datetime(2026, 8, 24, 9, 14, tzinfo=IST), Timeframe.M5
            )

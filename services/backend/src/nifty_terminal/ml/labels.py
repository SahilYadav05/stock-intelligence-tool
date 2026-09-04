"""Volatility-adjusted first-touch labels built strictly after decision time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Candle, CandleStatus, Timeframe
from nifty_terminal.features.models import PriceFeatureRow
from nifty_terminal.ml.definitions import (
    DOWN_ATR_MULTIPLIER,
    HORIZON_BARS,
    HORIZON_MINUTES,
    LABEL_DEFINITION_HASH,
    LABEL_VERSION,
    UP_ATR_MULTIPLIER,
)
from nifty_terminal.ml.models import FirstTouchLabel, TargetOutcome


@dataclass(frozen=True, slots=True)
class FirstTouchLabelConfig:
    label_version: str
    label_definition_hash: str
    horizon_bars: int
    up_atr_multiplier: Decimal
    down_atr_multiplier: Decimal

    @property
    def horizon_minutes(self) -> int:
        return self.horizon_bars * Timeframe.M5.minutes


DEFAULT_FIRST_TOUCH_CONFIG = FirstTouchLabelConfig(
    label_version=LABEL_VERSION,
    label_definition_hash=LABEL_DEFINITION_HASH,
    horizon_bars=HORIZON_BARS,
    up_atr_multiplier=UP_ATR_MULTIPLIER,
    down_atr_multiplier=DOWN_ATR_MULTIPLIER,
)


def symmetric_first_touch_config(multiplier: Decimal) -> FirstTouchLabelConfig:
    """Create a versioned 60-minute research target without changing the MVP target."""
    if multiplier <= 0:
        raise ValueError("ATR barrier multiplier must be positive")
    normalized = format(multiplier.normalize(), "f")
    definition = {
        "reference": "finalized 5m close at decision_time",
        "volatility": "Wilder ATR(14) from finalized 5m candles at decision_time",
        "up": f"+{normalized} ATR touched first",
        "down": f"-{normalized} ATR touched first",
        "neither": "neither barrier touched in 12 subsequent finalized 5m candles",
        "ambiguous": "both barriers touched before one-minute ordering can resolve",
        "session": "complete 60-minute window remains inside one NSE session",
        "purpose": "Step 15 screening only; not approved for live inference",
    }
    identity = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FirstTouchLabelConfig(
        label_version=f"nifty_5m_atr_first_touch.60m.symmetric_{normalized}.research.v1",
        label_definition_hash=identity,
        horizon_bars=HORIZON_BARS,
        up_atr_multiplier=multiplier,
        down_atr_multiplier=multiplier,
    )


class FirstTouchLabeler:
    """Creates audit-ready labels without allowing future data into features."""

    def __init__(
        self,
        calendar: NseSessionCalendar,
        config: FirstTouchLabelConfig | None = None,
    ) -> None:
        self._calendar = calendar
        self._config = config or DEFAULT_FIRST_TOUCH_CONFIG

    def build(
        self,
        *,
        dataset_id: str,
        primary_candles: tuple[Candle, ...],
        primary_features: tuple[PriceFeatureRow, ...],
        minute_candles: tuple[Candle, ...] = (),
    ) -> tuple[FirstTouchLabel, ...]:
        _validate_candles(primary_candles, Timeframe.M5)
        _validate_candles(minute_candles, Timeframe.M1)
        feature_by_candle = {item.source_candle_id: item for item in primary_features}
        future_by_open = {item.opens_at: item for item in primary_candles}
        minute_by_open = {item.opens_at: item for item in minute_candles}

        labels = [
            self._label_one(
                dataset_id=dataset_id,
                candle=candle,
                feature=feature_by_candle.get(candle.candle_id),
                future_by_open=future_by_open,
                minute_by_open=minute_by_open,
            )
            for candle in primary_candles
        ]
        return tuple(labels)

    def _label_one(
        self,
        *,
        dataset_id: str,
        candle: Candle,
        feature: PriceFeatureRow | None,
        future_by_open: dict[datetime, Candle],
        minute_by_open: dict[datetime, Candle],
    ) -> FirstTouchLabel:
        window_ends_at = candle.closes_at + timedelta(
            minutes=self._config.horizon_minutes
        )
        label_id = str(
            uuid5(
                NAMESPACE_URL,
                "label:"
                f"{dataset_id}:{candle.candle_id}:"
                f"{self._config.label_definition_hash}",
            )
        )
        atr_value = feature.get("atr_14") if feature else None
        atr = atr_value if isinstance(atr_value, Decimal) else None
        if atr is None or atr <= 0:
            return _excluded(
                label_id=label_id,
                dataset_id=dataset_id,
                candle=candle,
                window_ends_at=window_ends_at,
                reason="ATR_AT_DECISION_UNAVAILABLE",
                config=self._config,
            )

        up_barrier = candle.close + atr * self._config.up_atr_multiplier
        down_barrier = candle.close - atr * self._config.down_atr_multiplier
        session = self._calendar.session_containing(candle.opens_at)
        if session is None or window_ends_at > session.closes_at:
            return _excluded(
                label_id=label_id,
                dataset_id=dataset_id,
                candle=candle,
                window_ends_at=window_ends_at,
                reason="OUTCOME_WINDOW_CROSSES_SESSION_CLOSE",
                atr=atr,
                up_barrier=up_barrier,
                down_barrier=down_barrier,
                config=self._config,
            )

        future: list[Candle] = []
        for offset in range(self._config.horizon_bars):
            opens_at = candle.closes_at + timedelta(minutes=offset * Timeframe.M5.minutes)
            item = future_by_open.get(opens_at)
            if item is None:
                return _excluded(
                    label_id=label_id,
                    dataset_id=dataset_id,
                    candle=candle,
                    window_ends_at=window_ends_at,
                    reason="INCOMPLETE_FUTURE_5M_WINDOW",
                    atr=atr,
                    up_barrier=up_barrier,
                    down_barrier=down_barrier,
                    future=tuple(future),
                    config=self._config,
                )
            future.append(item)

        for item in future:
            up_hit = item.high >= up_barrier
            down_hit = item.low <= down_barrier
            if not up_hit and not down_hit:
                continue
            if up_hit and down_hit:
                resolution = _resolve_with_minutes(
                    five_minute=item,
                    up_barrier=up_barrier,
                    down_barrier=down_barrier,
                    minute_by_open=minute_by_open,
                )
                if resolution is None:
                    return _completed(
                        label_id=label_id,
                        dataset_id=dataset_id,
                        candle=candle,
                        atr=atr,
                        up_barrier=up_barrier,
                        down_barrier=down_barrier,
                        window_ends_at=window_ends_at,
                        future=tuple(future),
                        outcome=TargetOutcome.AMBIGUOUS,
                        first_touch_at=item.closes_at,
                        first_touch_candle_id=item.candle_id,
                        eligible=False,
                        exclusion_reason="AMBIGUOUS_INTRABAR_ORDER",
                        config=self._config,
                    )
                outcome, touch = resolution
                return _completed(
                    label_id=label_id,
                    dataset_id=dataset_id,
                    candle=candle,
                    atr=atr,
                    up_barrier=up_barrier,
                    down_barrier=down_barrier,
                    window_ends_at=window_ends_at,
                    future=tuple(future),
                    outcome=outcome,
                    first_touch_at=touch.closes_at,
                    first_touch_candle_id=touch.candle_id,
                    eligible=True,
                    config=self._config,
                )
            outcome = TargetOutcome.UP if up_hit else TargetOutcome.DOWN
            return _completed(
                label_id=label_id,
                dataset_id=dataset_id,
                candle=candle,
                atr=atr,
                up_barrier=up_barrier,
                down_barrier=down_barrier,
                window_ends_at=window_ends_at,
                future=tuple(future),
                outcome=outcome,
                first_touch_at=item.closes_at,
                first_touch_candle_id=item.candle_id,
                eligible=True,
                config=self._config,
            )

        return _completed(
            label_id=label_id,
            dataset_id=dataset_id,
            candle=candle,
            atr=atr,
            up_barrier=up_barrier,
            down_barrier=down_barrier,
            window_ends_at=window_ends_at,
            future=tuple(future),
            outcome=TargetOutcome.NEITHER,
            first_touch_at=None,
            first_touch_candle_id=None,
            eligible=True,
            config=self._config,
        )


def _resolve_with_minutes(
    *,
    five_minute: Candle,
    up_barrier: Decimal,
    down_barrier: Decimal,
    minute_by_open: dict[datetime, Candle],
) -> tuple[TargetOutcome, Candle] | None:
    minutes: list[Candle] = []
    for offset in range(Timeframe.M5.minutes):
        opens_at = five_minute.opens_at + timedelta(minutes=offset)
        minute = minute_by_open.get(opens_at)
        if minute is None:
            return None
        minutes.append(minute)
    for minute in minutes:
        up_hit = minute.high >= up_barrier
        down_hit = minute.low <= down_barrier
        if up_hit and down_hit:
            return None
        if up_hit:
            return TargetOutcome.UP, minute
        if down_hit:
            return TargetOutcome.DOWN, minute
    return None


def _validate_candles(candles: tuple[Candle, ...], timeframe: Timeframe) -> None:
    if not candles:
        return
    if any(item.timeframe is not timeframe for item in candles):
        raise ValueError(f"Expected only {timeframe.value} candles")
    if any(item.status is not CandleStatus.FINALIZED for item in candles):
        raise ValueError("Labels require finalized candles")
    if any(item.instrument_id != candles[0].instrument_id for item in candles):
        raise ValueError("Label inputs cannot mix instruments")
    if tuple(item.opens_at for item in candles) != tuple(
        sorted(item.opens_at for item in candles)
    ):
        raise ValueError("Label candles must be chronological")


def _excluded(
    *,
    label_id: str,
    dataset_id: str,
    candle: Candle,
    window_ends_at: datetime,
    reason: str,
    atr: Decimal | None = None,
    up_barrier: Decimal | None = None,
    down_barrier: Decimal | None = None,
    future: tuple[Candle, ...] = (),
    config: FirstTouchLabelConfig = DEFAULT_FIRST_TOUCH_CONFIG,
) -> FirstTouchLabel:
    return FirstTouchLabel(
        schema_version=1,
        label_id=label_id,
        label_version=config.label_version,
        label_definition_hash=config.label_definition_hash,
        dataset_id=dataset_id,
        instrument_id=candle.instrument_id,
        decision_candle_id=candle.candle_id,
        decision_time=candle.closes_at,
        reference_close=candle.close,
        atr_at_decision=atr,
        up_barrier=up_barrier,
        down_barrier=down_barrier,
        window_ends_at=window_ends_at,
        outcome=None,
        first_touch_at=None,
        first_touch_candle_id=None,
        future_candle_ids=tuple(item.candle_id for item in future),
        eligible=False,
        exclusion_reason=reason,
    )


def _completed(
    *,
    label_id: str,
    dataset_id: str,
    candle: Candle,
    atr: Decimal,
    up_barrier: Decimal,
    down_barrier: Decimal,
    window_ends_at: datetime,
    future: tuple[Candle, ...],
    outcome: TargetOutcome,
    first_touch_at: datetime | None,
    first_touch_candle_id: str | None,
    eligible: bool,
    exclusion_reason: str | None = None,
    config: FirstTouchLabelConfig = DEFAULT_FIRST_TOUCH_CONFIG,
) -> FirstTouchLabel:
    return FirstTouchLabel(
        schema_version=1,
        label_id=label_id,
        label_version=config.label_version,
        label_definition_hash=config.label_definition_hash,
        dataset_id=dataset_id,
        instrument_id=candle.instrument_id,
        decision_candle_id=candle.candle_id,
        decision_time=candle.closes_at,
        reference_close=candle.close,
        atr_at_decision=atr,
        up_barrier=up_barrier,
        down_barrier=down_barrier,
        window_ends_at=window_ends_at,
        outcome=outcome,
        first_touch_at=first_touch_at,
        first_touch_candle_id=first_touch_candle_id,
        future_candle_ids=tuple(item.candle_id for item in future),
        eligible=eligible,
        exclusion_reason=exclusion_reason,
    )

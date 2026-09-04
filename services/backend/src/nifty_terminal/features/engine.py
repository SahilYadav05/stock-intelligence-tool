"""Pure batch feature function used identically by history and live snapshots."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, localcontext

from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.domain.candle import Candle, CandleStatus, Timeframe
from nifty_terminal.features.definitions import (
    FEATURE_SET_HASH,
    FEATURE_VERSION,
    MINIMUM_HISTORY,
)
from nifty_terminal.features.models import FeatureValue, PriceFeatureRow


ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")


class PriceFeatureEngine:
    """Calculates causal price features from finalized candles only."""

    def __init__(self, calendar: NseSessionCalendar) -> None:
        self._calendar = calendar

    def calculate(self, candles: tuple[Candle, ...]) -> tuple[PriceFeatureRow, ...]:
        if not candles:
            return ()
        _validate_series(candles)
        closes = [item.close for item in candles]
        returns = [None] + [_ratio(closes[index], closes[index - 1]) for index in range(1, len(candles))]
        true_ranges = _true_ranges(candles)
        sma20 = _rolling_mean(closes, 20)
        sma50 = _rolling_mean(closes, 50)
        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        atr14 = _wilder_average(true_ranges, 14)
        rsi14 = _rsi(closes, 14)
        rolling_vol20 = _rolling_std_optional(returns, 20)
        close_std20 = _rolling_std(closes, 20)
        unexpected_gap = _unexpected_gap_flags(candles)

        rows: list[PriceFeatureRow] = []
        with localcontext() as context:
            context.prec = 34
            for index, candle in enumerate(candles):
                atr = atr14[index]
                values: tuple[tuple[str, FeatureValue], ...] = (
                    ("return_1", returns[index]),
                    ("return_5", _period_return(closes, index, 5)),
                    ("log_return_1", _log_return(closes, index)),
                    ("range_pct", (candle.high - candle.low) / candle.close),
                    ("body_pct", (candle.close - candle.open) / candle.open),
                    ("upper_wick_pct", (candle.high - max(candle.open, candle.close)) / candle.open),
                    ("lower_wick_pct", (min(candle.open, candle.close) - candle.low) / candle.open),
                    ("sma_20", sma20[index]),
                    ("sma_50", sma50[index]),
                    ("ema_20", ema20[index]),
                    ("ema_50", ema50[index]),
                    ("atr_14", atr),
                    ("atr_pct", atr / candle.close if atr is not None else None),
                    ("rsi_14", rsi14[index]),
                    ("rolling_vol_20", rolling_vol20[index]),
                    (
                        "bollinger_z_20",
                        _safe_div(candle.close - sma20[index], close_std20[index])
                        if sma20[index] is not None
                        else None,
                    ),
                    ("roc_5", _period_return(closes, index, 5)),
                    ("roc_12", _period_return(closes, index, 12)),
                    (
                        "distance_ema20_atr",
                        _safe_div(candle.close - ema20[index], atr)
                        if ema20[index] is not None
                        else None,
                    ),
                    ("range_atr", _safe_div(candle.high - candle.low, atr)),
                    (
                        "trend_ema20_above_ema50",
                        ema20[index] > ema50[index]
                        if ema20[index] is not None and ema50[index] is not None
                        else None,
                    ),
                    ("breakout_up_20", _breakout_up(candles, index, 20)),
                    ("breakout_down_20", _breakout_down(candles, index, 20)),
                    ("minute_of_session", _minute_of_session(self._calendar, candle)),
                    ("minutes_to_session_close", _minutes_to_close(self._calendar, candle)),
                    ("day_of_week", candle.opens_at.astimezone(IST).weekday()),
                )
                blockers: list[str] = []
                if index + 1 < MINIMUM_HISTORY:
                    blockers.append(f"INSUFFICIENT_HISTORY:{index + 1}/{MINIMUM_HISTORY}")
                if any(unexpected_gap[max(0, index - MINIMUM_HISTORY + 1) : index + 1]):
                    blockers.append("INTRADAY_GAP_IN_FEATURE_WINDOW")
                if any(value is None for _, value in values):
                    blockers.append("FEATURE_WARMUP_INCOMPLETE")
                rows.append(
                    PriceFeatureRow(
                        schema_version=1,
                        feature_version=FEATURE_VERSION,
                        feature_set_hash=FEATURE_SET_HASH,
                        source_candle_id=candle.candle_id,
                        instrument_id=candle.instrument_id,
                        timeframe=candle.timeframe,
                        decision_time=candle.closes_at,
                        values=values,
                        is_ready=not blockers,
                        blockers=tuple(blockers),
                    )
                )
        return tuple(rows)


def _validate_series(candles: tuple[Candle, ...]) -> None:
    first = candles[0]
    keys = [(item.opens_at, item.revision) for item in candles]
    if keys != sorted(keys) or len({item.opens_at for item in candles}) != len(candles):
        raise ValueError("Feature candles must be chronological latest revisions")
    if any(item.status is not CandleStatus.FINALIZED for item in candles):
        raise ValueError("Developing candles cannot enter the feature engine")
    if any(item.instrument_id != first.instrument_id for item in candles):
        raise ValueError("Feature series cannot mix instruments")
    if any(item.timeframe is not first.timeframe for item in candles):
        raise ValueError("Feature series cannot mix timeframes")


def _ratio(current: Decimal, previous: Decimal) -> Decimal:
    return current / previous - ONE


def _period_return(values: list[Decimal], index: int, period: int) -> Decimal | None:
    return _ratio(values[index], values[index - period]) if index >= period else None


def _log_return(values: list[Decimal], index: int) -> Decimal | None:
    if index == 0:
        return None
    return (values[index] / values[index - 1]).ln()


def _rolling_mean(values: list[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(values)
    running = ZERO
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / Decimal(period)
    return result


def _ema(values: list[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(values)
    if len(values) < period:
        return result
    seed = sum(values[:period], ZERO) / Decimal(period)
    result[period - 1] = seed
    alpha = Decimal(2) / Decimal(period + 1)
    previous = seed
    for index in range(period, len(values)):
        previous = alpha * values[index] + (ONE - alpha) * previous
        result[index] = previous
    return result


def _true_ranges(candles: tuple[Candle, ...]) -> list[Decimal]:
    result: list[Decimal] = []
    for index, candle in enumerate(candles):
        if index == 0:
            result.append(candle.high - candle.low)
        else:
            previous_close = candles[index - 1].close
            result.append(
                max(
                    candle.high - candle.low,
                    abs(candle.high - previous_close),
                    abs(candle.low - previous_close),
                )
            )
    return result


def _wilder_average(values: list[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(values)
    if len(values) < period:
        return result
    previous = sum(values[:period], ZERO) / Decimal(period)
    result[period - 1] = previous
    for index in range(period, len(values)):
        previous = (previous * Decimal(period - 1) + values[index]) / Decimal(period)
        result[index] = previous
    return result


def _rsi(values: list[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(values)
    if len(values) <= period:
        return result
    changes = [values[index] - values[index - 1] for index in range(1, len(values))]
    average_gain = sum((max(item, ZERO) for item in changes[:period]), ZERO) / Decimal(period)
    average_loss = sum((max(-item, ZERO) for item in changes[:period]), ZERO) / Decimal(period)
    result[period] = _rsi_value(average_gain, average_loss)
    for index in range(period + 1, len(values)):
        change = changes[index - 1]
        average_gain = (average_gain * Decimal(period - 1) + max(change, ZERO)) / Decimal(period)
        average_loss = (average_loss * Decimal(period - 1) + max(-change, ZERO)) / Decimal(period)
        result[index] = _rsi_value(average_gain, average_loss)
    return result


def _rsi_value(gain: Decimal, loss: Decimal) -> Decimal:
    if loss == ZERO:
        return HUNDRED if gain > ZERO else Decimal("50")
    return HUNDRED - HUNDRED / (ONE + gain / loss)


def _rolling_std(values: list[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        result[index] = _population_std(values[index - period + 1 : index + 1])
    return result


def _rolling_std_optional(
    values: list[Decimal | None], period: int
) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(values)
    for index in range(period, len(values)):
        window = values[index - period + 1 : index + 1]
        if all(item is not None for item in window):
            result[index] = _population_std([item for item in window if item is not None])
    return result


def _population_std(values: list[Decimal]) -> Decimal:
    mean = sum(values, ZERO) / Decimal(len(values))
    variance = sum(((item - mean) ** 2 for item in values), ZERO) / Decimal(len(values))
    return variance.sqrt()


def _safe_div(numerator: Decimal, denominator: Decimal | None) -> Decimal | None:
    return numerator / denominator if denominator not in {None, ZERO} else None


def _breakout_up(candles: tuple[Candle, ...], index: int, period: int) -> bool | None:
    if index < period:
        return None
    return candles[index].close > max(item.high for item in candles[index - period : index])


def _breakout_down(candles: tuple[Candle, ...], index: int, period: int) -> bool | None:
    if index < period:
        return None
    return candles[index].close < min(item.low for item in candles[index - period : index])


def _unexpected_gap_flags(candles: tuple[Candle, ...]) -> list[bool]:
    flags = [False] * len(candles)
    for index in range(1, len(candles)):
        previous = candles[index - 1]
        current = candles[index]
        same_session_date = (
            previous.opens_at.astimezone(IST).date()
            == current.opens_at.astimezone(IST).date()
        )
        flags[index] = same_session_date and current.opens_at != previous.closes_at
    return flags


def _minute_of_session(calendar: NseSessionCalendar, candle: Candle) -> int:
    session = calendar.session_containing(candle.opens_at)
    if session is None:
        raise ValueError("Feature candle is outside the configured NSE session")
    return int((candle.opens_at.astimezone(IST) - session.opens_at).total_seconds() // 60)


def _minutes_to_close(calendar: NseSessionCalendar, candle: Candle) -> int:
    session = calendar.session_containing(candle.opens_at)
    if session is None:
        raise ValueError("Feature candle is outside the configured NSE session")
    return int((session.closes_at - candle.closes_at.astimezone(IST)).total_seconds() // 60)

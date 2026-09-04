"""Acquire, audit, and persist long-range Angel One NIFTY research history."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.features.engine import PriceFeatureEngine
from nifty_terminal.features.materializer import HistoricalFeatureMaterializer
from nifty_terminal.history.calendar_loader import (
    load_nse_calendar,
    validate_calendar_coverage,
)
from nifty_terminal.history.models import HistoricalRequest, QualityStatus
from nifty_terminal.history.pipeline import HistoricalImportPipeline
from nifty_terminal.history.session_normalizer import (
    diagnose_expected_minute_coverage,
    normalize_to_continuous_sessions,
)
from nifty_terminal.history.sources.angelone_source import (
    AcquiredHistoricalDataSource,
    AngelOneHistoricalAcquirer,
)
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.providers.angelone import (
    AngelOneProviderError,
    build_angelone_adapter,
)
from nifty_terminal.settings import Settings


MVP_INSTRUMENT_ID = "NIFTY50_SPOT"
MINIMUM_RESEARCH_SAMPLES = 5_000
MINIMUM_CLASS_SAMPLES = 250
DEFAULT_CALENDAR = Path("config/nse-calendar-through-2026-08-25.json")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--to-date", type=date.fromisoformat)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--request-delay-ms", type=int, default=400)
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=Path("artifacts/research/history"),
    )
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> int:
    if not arguments.env_file.is_file():
        raise ValueError(f"Environment file not found: {arguments.env_file}")
    if not arguments.calendar_json.is_file():
        raise ValueError(f"Exchange calendar file not found: {arguments.calendar_json}")
    load_dotenv(arguments.env_file, override=False)
    settings = Settings.from_environment()
    if not settings.live_signal_kill_switch:
        raise ValueError("LIVE_SIGNAL_KILL_SWITCH must remain true during research acquisition")

    calendar_metadata = json.loads(arguments.calendar_json.read_text(encoding="utf-8"))
    calendar = load_nse_calendar(arguments.calendar_json)
    now = datetime.now(timezone.utc)
    to_date = arguments.to_date or _latest_completed_session_date(calendar, now)
    validate_calendar_coverage(
        calendar_metadata,
        starts_on=arguments.from_date,
        ends_on=to_date,
    )
    if arguments.from_date > to_date:
        raise ValueError("--from-date must not be later than --to-date")

    request = HistoricalRequest(
        instrument_id=MVP_INSTRUMENT_ID,
        timeframe=Timeframe.M1,
        starts_at=datetime.combine(arguments.from_date, time.min, IST).astimezone(timezone.utc),
        ends_at=datetime.combine(
            to_date + timedelta(days=1),
            time.min,
            IST,
        ).astimezone(timezone.utc),
    )
    adapter = build_angelone_adapter(settings)
    acquirer = AngelOneHistoricalAcquirer(
        provider=adapter,
        provider_name=adapter.provider_name,
        chunk_days=arguments.chunk_days,
        request_delay_milliseconds=arguments.request_delay_ms,
    )

    print("Step 13 Angel One historical research acquisition", file=sys.stderr)
    print(f"- range: {arguments.from_date} through {to_date} IST", file=sys.stderr)
    print("- source timeframe: finalized 1m", file=sys.stderr)
    print("- NIFTY spot volume: null", file=sys.stderr)
    print("- signals and order execution: disabled", file=sys.stderr)

    await adapter.connect()
    try:
        raw_batch = await acquirer.acquire(request, progress=_progress)
    finally:
        await adapter.disconnect()

    normalization = normalize_to_continuous_sessions(
        raw_batch,
        calendar=calendar,
        calendar_metadata=calendar_metadata,
    )
    batch = normalization.batch
    coverage = diagnose_expected_minute_coverage(
        batch,
        calendar=calendar,
        calendar_metadata=calendar_metadata,
    )
    print(
        f"- session normalization: retained {len(batch.rows)}; "
        f"excluded {normalization.excluded_row_count} explicitly authorized "
        "non-trading observations",
        file=sys.stderr,
    )
    print(
        f"- expected-minute coverage: {coverage.coverage_ratio * 100:.6f}%",
        file=sys.stderr,
    )

    repository = SQLiteHistoricalRepository(arguments.database)
    result = HistoricalImportPipeline(
        source=AcquiredHistoricalDataSource(batch),
        repository=repository,
        calendar=calendar,
        registry=build_mvp_instrument_registry(),
    ).run(request)

    feature_rows: dict[str, int] = {}
    candle_counts: dict[str, int] = {}
    candles_by_timeframe = {}
    if result.quality.status is not QualityStatus.REJECTED:
        materializer = HistoricalFeatureMaterializer(
            repository=repository,
            engine=PriceFeatureEngine(calendar),
        )
        for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1):
            candles = repository.load_latest_candles(
                dataset_id=result.dataset_id,
                instrument_id=MVP_INSTRUMENT_ID,
                timeframe=timeframe,
            )
            candles_by_timeframe[timeframe] = candles
            candle_counts[timeframe.value] = len(candles)
            if timeframe is not Timeframe.M1:
                materializer.run(
                    dataset_id=result.dataset_id,
                    instrument_id=MVP_INSTRUMENT_ID,
                    timeframe=timeframe,
                )
                feature_rows[timeframe.value] = repository.count_feature_rows(
                    dataset_id=result.dataset_id,
                    feature_version="price_features.v1",
                    timeframe=timeframe,
                )

    dataset_summary = None
    readiness_blockers: list[str] = []
    accepted_warnings = set(result.quality.warnings).issubset({"INTRADAY_MINUTE_GAPS"})
    research_quality_accepted = (
        result.quality.status is QualityStatus.PASS
        or (
            result.quality.status is QualityStatus.DEGRADED
            and not result.quality.errors
            and accepted_warnings
            and coverage.research_acceptable
        )
    )
    if not research_quality_accepted:
        readiness_blockers.append(f"HISTORICAL_QUALITY_{result.quality.status.value}")
        readiness_blockers.extend(coverage.blockers())
    required = (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1)
    if any(not candles_by_timeframe.get(timeframe) for timeframe in required):
        readiness_blockers.append("REQUIRED_CANONICAL_TIMEFRAME_MISSING")
    if not readiness_blockers:
        dataset_report = TrainingDatasetAssembler(calendar).assemble(
            dataset_id=result.dataset_id,
            minute_candles=candles_by_timeframe[Timeframe.M1],
            primary_candles=candles_by_timeframe[Timeframe.M5],
            context_15m_candles=candles_by_timeframe[Timeframe.M15],
            context_1h_candles=candles_by_timeframe[Timeframe.H1],
        )
        dataset_summary = dataset_report.summary_contract()
        if dataset_report.eligible_samples < MINIMUM_RESEARCH_SAMPLES:
            readiness_blockers.append(
                f"ELIGIBLE_SAMPLES_BELOW_{MINIMUM_RESEARCH_SAMPLES}"
            )
        for outcome, count in dataset_report.outcome_support:
            if count < MINIMUM_CLASS_SAMPLES:
                readiness_blockers.append(
                    f"{outcome}_SAMPLES_BELOW_{MINIMUM_CLASS_SAMPLES}"
                )

    payload = {
        "schema_version": 1,
        "step": 13,
        "instrument_id": MVP_INSTRUMENT_ID,
        "provider": "angelone",
        "requested_from": arguments.from_date.isoformat(),
        "requested_through": to_date.isoformat(),
        "calendar_file": str(arguments.calendar_json),
        "calendar_verified_through": calendar_metadata["verified_through"],
        "closing_auction_policy": {
            "effective_from": "2026-08-03",
            "continuous_candles_end": "15:15 IST",
            "auction_observations_are_model_inputs": False,
            "standard_60m_signal_last_decision": "14:15 IST",
        },
        "dataset_id": result.dataset_id,
        "dataset_imported": result.imported,
        "raw_source_sha256": normalization.raw_source_sha256,
        "source_sha256": batch.source_sha256,
        "raw_source_minute_rows": normalization.raw_row_count,
        "source_minute_rows": len(batch.rows),
        "session_normalization": normalization.to_contract(),
        "expected_minute_coverage": coverage.to_contract(),
        "research_quality_accepted": research_quality_accepted,
        "candle_revision_count": result.candle_revision_count,
        "canonical_candle_counts": candle_counts,
        "materialized_feature_rows": feature_rows,
        "quality": result.quality.to_contract(),
        "training_dataset_summary": dataset_summary,
        "training_research_ready": not readiness_blockers,
        "training_readiness_blockers": sorted(set(readiness_blockers)),
        "approved_for_live_inference": False,
        "calibrated_probabilities_available": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "nifty_spot_volume_enabled": False,
    }
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_directory / f"{result.dataset_id}.json"
    payload["report_path"] = str(report_path)
    report_written = _write_immutable_report(report_path, payload)
    payload["report_written"] = report_written
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not readiness_blockers else 2


def _progress(completed: int, total: int, rows: int) -> None:
    if completed == 1 or completed == total or completed % 10 == 0:
        print(
            f"- historical chunks: {completed}/{total}; unique rows: {rows}",
            file=sys.stderr,
        )


def _latest_completed_session_date(
    calendar: NseSessionCalendar,
    now: datetime,
) -> date:
    local = now.astimezone(IST)
    candidate = local.date()
    session = calendar.session_for_date(candidate)
    if session is not None and local >= session.closes_at:
        return candidate
    candidate -= timedelta(days=1)
    while calendar.session_for_date(candidate) is None:
        candidate -= timedelta(days=1)
    return candidate


def _write_immutable_report(path: Path, payload: dict[str, object]) -> bool:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(rendered)
        return True
    except FileExistsError:
        return False


def main() -> int:
    try:
        return asyncio.run(_run(_arguments()))
    except (AngelOneProviderError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"RESULT: STEP 13 ACQUISITION FAILED: {error}")
        print("No model, signal, or order was created.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

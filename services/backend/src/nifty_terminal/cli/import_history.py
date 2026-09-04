"""Import an authorized provider CSV into the immutable local research store."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.features.engine import PriceFeatureEngine
from nifty_terminal.features.materializer import HistoricalFeatureMaterializer
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.models import HistoricalRequest, QualityStatus
from nifty_terminal.history.pipeline import HistoricalImportPipeline
from nifty_terminal.history.sources.csv_source import CsvHistoricalDataSource
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--starts-at", required=True, type=_timestamp)
    parser.add_argument("--ends-at", required=True, type=_timestamp)
    parser.add_argument("--calendar-json", type=Path)
    arguments = parser.parse_args()

    calendar = load_nse_calendar(arguments.calendar_json)
    repository = SQLiteHistoricalRepository(arguments.database)
    result = HistoricalImportPipeline(
        source=CsvHistoricalDataSource(path=arguments.csv, provider=arguments.provider),
        repository=repository,
        calendar=calendar,
        registry=build_mvp_instrument_registry(),
    ).run(
        HistoricalRequest(
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M1,
            starts_at=arguments.starts_at,
            ends_at=arguments.ends_at,
        )
    )

    feature_rows: dict[str, int] = {}
    if result.imported and result.quality.status is not QualityStatus.REJECTED:
        materializer = HistoricalFeatureMaterializer(
            repository=repository,
            engine=PriceFeatureEngine(calendar),
        )
        for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
            feature_rows[timeframe.value] = materializer.run(
                dataset_id=result.dataset_id,
                instrument_id="NIFTY50_SPOT",
                timeframe=timeframe,
            )

    print(
        json.dumps(
            {
                "dataset_id": result.dataset_id,
                "imported": result.imported,
                "candle_revision_count": result.candle_revision_count,
                "quality": result.quality.to_contract(),
                "materialized_feature_rows": feature_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("Timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    main()

"""Run Step 15 nested target and probability-model screening on PASS history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.cli.run_real_data_research import (
    DEFAULT_CALENDAR,
    _require_live_signal_kill_switch,
    _write_immutable_json,
)
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.labels import symmetric_first_touch_config
from nifty_terminal.research.v2 import (
    BARRIER_MULTIPLIERS,
    RESEARCH_V2_IDENTITY,
    ResearchV2Report,
    screen_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/research/probability-v2"),
    )
    arguments = parser.parse_args()

    _require_live_signal_kill_switch(arguments.env_file)
    if not arguments.calendar_json.is_file():
        raise SystemExit(
            f"Sourced exchange calendar does not exist: {arguments.calendar_json}"
        )
    repository = SQLiteHistoricalRepository(arguments.database)
    dataset_id = arguments.dataset_id or repository.latest_pass_dataset_id(
        instrument_id="NIFTY50_SPOT"
    )
    if dataset_id is None:
        raise SystemExit("No immutable PASS NIFTY50_SPOT dataset is available")
    calendar = load_nse_calendar(arguments.calendar_json)
    candles = {
        timeframe: repository.load_latest_candles(
            dataset_id=dataset_id,
            instrument_id="NIFTY50_SPOT",
            timeframe=timeframe,
        )
        for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1)
    }
    if any(not rows for rows in candles.values()):
        raise SystemExit("The PASS dataset is missing a required canonical timeframe")

    print("Step 15 probability and target research v2")
    print(f"- dataset: {dataset_id}")
    print(f"- sourced exchange calendar: {arguments.calendar_json}")
    print("- symmetric 60m barriers: 1.0, 1.25 and 1.5 ATR")
    print("- candidates: logistic/HGB; unweighted and balanced")
    print("- nested chronology: folds 0-2 selection, 3 calibration, 4 final screening")
    print("- this is screening, not a live-model release")
    print("- local CPU runtime may be 10-30 minutes")

    targets = []
    for multiplier in BARRIER_MULTIPLIERS:
        print(f"- screening symmetric {format(multiplier, 'f')} ATR target...", flush=True)
        config = symmetric_first_touch_config(multiplier)
        dataset = TrainingDatasetAssembler(calendar, label_config=config).assemble(
            dataset_id=dataset_id,
            minute_candles=candles[Timeframe.M1],
            primary_candles=candles[Timeframe.M5],
            context_15m_candles=candles[Timeframe.M15],
            context_1h_candles=candles[Timeframe.H1],
        )
        targets.append(screen_target(dataset, multiplier))

    report = ResearchV2Report(
        dataset_id=dataset_id,
        feature_version=FEATURE_VERSION,
        feature_set_hash=FEATURE_SET_HASH,
        targets=tuple(targets),
    )
    created_at = datetime.now(timezone.utc)
    experiment_id = str(
        uuid5(
            NAMESPACE_URL,
            f"probability-v2:{dataset_id}:{RESEARCH_V2_IDENTITY}:{created_at.isoformat()}",
        )
    )
    payload = report.to_contract()
    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / f"{experiment_id}.json"
    _write_immutable_json(output_path, payload)
    printable = dict(payload)
    printable["report_path"] = str(output_path)
    print(json.dumps(printable, indent=2, sort_keys=True))
    print("RESULT: STEP 15 PROBABILITY RESEARCH COMPLETED")
    print("No model, precise live probability, official signal or order was released.")


if __name__ == "__main__":
    main()

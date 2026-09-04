"""Run Step 18 enhanced-feature and hierarchical-model research."""

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
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.labels import symmetric_first_touch_config
from nifty_terminal.research.step16 import LOCKED_ATR_MULTIPLIER
from nifty_terminal.research.step18 import (
    CANDIDATE_NAMES,
    RESEARCH_IDENTITY,
    STEP18_VERSION,
    run_model_v2_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--report-dir", type=Path, default=Path("artifacts/research/model-v2")
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

    print("Step 18 hierarchical model-v2 research")
    print(f"- dataset: {dataset_id}")
    print(f"- sourced exchange calendar: {arguments.calendar_json}")
    print("- locked target: symmetric 1.5 ATR first-touch over 60 minutes")
    print("- enhanced inputs: finalized 5m, 15m and 1h price features only")
    print(f"- candidates: {', '.join(CANDIDATE_NAMES)}")
    print("- chronology: folds 0-1 architecture, 2 calibration fit, 3 calibration selection")
    print("- fold 4: historical diagnostics only; forward confirmation still mandatory")
    print("- anti-collapse gates require both UP and DOWN prediction support")
    print("- Step 17 shadow manifest and ledger will not be modified")
    print("- official signals and automatic trading remain disabled")
    print("- local CPU runtime may be 10-30 minutes")

    dataset = TrainingDatasetAssembler(
        load_nse_calendar(arguments.calendar_json),
        label_config=symmetric_first_touch_config(LOCKED_ATR_MULTIPLIER),
    ).assemble(
        dataset_id=dataset_id,
        minute_candles=candles[Timeframe.M1],
        primary_candles=candles[Timeframe.M5],
        context_15m_candles=candles[Timeframe.M15],
        context_1h_candles=candles[Timeframe.H1],
    )
    payload = run_model_v2_research(dataset)
    created_at = datetime.now(timezone.utc)
    experiment_id = str(
        uuid5(
            NAMESPACE_URL,
            f"step18:{dataset_id}:{RESEARCH_IDENTITY}:{created_at.isoformat()}",
        )
    )
    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_dir / f"{experiment_id}.json"
    _write_immutable_json(report_path, payload)

    diagnostic = dict(payload["historical_diagnostic"])
    printable = {
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "feature_architecture": payload["feature_architecture"],
        "selected_candidate": payload["selected_candidate"],
        "selected_candidate_selection_viable": payload[
            "selected_candidate_selection_viable"
        ],
        "selected_calibration_method": payload["selected_calibration_method"],
        "historical_diagnostic": diagnostic,
        "forward_confirmation": payload["forward_confirmation"],
        "existing_step17_runtime_modified": False,
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "report_path": str(report_path),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    print("RESULT: STEP 18 MODEL-V2 RESEARCH COMPLETED")
    print("No model or signal was released; inspect the research gates before Step 19.")


if __name__ == "__main__":
    main()

"""Run Step 14 chronological research and calibration on the latest PASS dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from nifty_terminal.calibration.research import Step7ResearchPipeline
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.pipeline import MLResearchPipeline
from nifty_terminal.research.real_data import (
    RealDataResearchResult,
    evaluate_real_data_research,
)


DEFAULT_CALENDAR = Path("config/nse-calendar-through-2026-08-25.json")
REAL_DATA_WALK_FORWARD_CONFIG = WalkForwardConfig(
    n_splits=5,
    minimum_train_samples=10_000,
    test_samples=2_000,
    purge_bars=12,
    embargo_bars=12,
    minimum_train_class_samples=25,
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
        default=Path("artifacts/research/real-data"),
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
        raise SystemExit(
            "No PASS NIFTY50_SPOT historical dataset exists. Complete Step 13.3 first."
        )

    print("Step 14 real-data chronological research")
    print(f"- dataset: {dataset_id}")
    print(f"- sourced exchange calendar: {arguments.calendar_json}")
    print("- target: UP / DOWN / NEITHER; 60-minute ATR first-touch")
    print("- validation: five purged chronological folds; 10,000 OOS predictions")
    print("- developing candles, NIFTY volume, VWAP and news: excluded")
    print("- live signals and automatic trading: disabled")
    print("- training may take several minutes on a local CPU")

    timestamp = datetime.now(timezone.utc)
    training_result = MLResearchPipeline(
        repository=repository,
        calendar=load_nse_calendar(arguments.calendar_json),
    ).run(
        dataset_id=dataset_id,
        instrument_id="NIFTY50_SPOT",
        config=REAL_DATA_WALK_FORWARD_CONFIG,
        created_at=timestamp,
    )
    observations, replay_inputs = repository.load_step7_source(
        run_id=training_result.report.run_id
    )
    step7 = Step7ResearchPipeline().run(
        observations=observations,
        replay_inputs=replay_inputs,
        created_at=timestamp,
    )
    calibration_persisted = repository.save_step7_research_report(step7)
    gate = evaluate_real_data_research(
        training=training_result.report,
        calibration=step7.calibration,
    )
    result = RealDataResearchResult(
        training=training_result.report,
        calibration=step7.calibration,
        gate=gate,
        training_persisted=training_result.persisted,
        calibration_persisted=calibration_persisted,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / f"{training_result.report.run_id}.json"
    _write_immutable_json(output_path, result.summary_contract())

    payload = result.summary_contract()
    payload["report_path"] = str(output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("RESULT: STEP 14 RESEARCH COMPLETED")
    if gate.passed:
        print("The research gate passed, but live inference remains disabled.")
    else:
        print("The research gate failed safely; do not tune thresholds to force a pass.")


def _require_live_signal_kill_switch(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Environment file does not exist: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    if values.get("LIVE_SIGNAL_KILL_SWITCH", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise SystemExit(
            "Step 14 requires LIVE_SIGNAL_KILL_SWITCH=true. Research cannot enable live signals."
        )


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(serialized)
    except FileExistsError:
        raise SystemExit(f"Refusing to overwrite immutable Step 14 report: {path}") from None


if __name__ == "__main__":
    main()

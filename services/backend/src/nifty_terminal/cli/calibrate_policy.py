"""Calibrate one Step 6 run and replay the deterministic Step 7 WAIT policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nifty_terminal.calibration.research import Step7ResearchPipeline
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration"))
    arguments = parser.parse_args()

    repository = SQLiteHistoricalRepository(arguments.database)
    observations, replay_inputs = repository.load_step7_source(run_id=arguments.run_id)
    report = Step7ResearchPipeline().run(
        observations=observations,
        replay_inputs=replay_inputs,
    )
    persisted = repository.save_step7_research_report(report)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / f"{report.calibration.calibration_id}.json"
    payload = json.dumps(report.to_contract(), indent=2, sort_keys=True) + "\n"
    try:
        with output_path.open("x", encoding="utf-8") as file:
            file.write(payload)
    except FileExistsError:
        raise SystemExit(
            f"Refusing to overwrite immutable Step 7 report: {output_path}"
        ) from None

    print(
        json.dumps(
            {
                "calibration_id": report.calibration.calibration_id,
                "persisted": persisted,
                "report_path": str(output_path),
                "release_gate_passed": report.calibration.release_gate_passed,
                "blockers": list(report.calibration.blockers),
                "signal_counts": report.to_contract()["policy"]["direction_support"],
                "automatic_execution": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

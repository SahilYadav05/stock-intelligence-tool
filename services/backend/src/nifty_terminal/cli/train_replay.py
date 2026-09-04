"""Train uncalibrated baselines and persist chronological simulated-live replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.pipeline import MLResearchPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--calendar-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--minimum-train-samples", type=int, default=1_500)
    parser.add_argument("--test-samples", type=int, default=250)
    parser.add_argument("--purge-bars", type=int, default=12)
    parser.add_argument("--embargo-bars", type=int, default=12)
    parser.add_argument("--minimum-train-class-samples", type=int, default=25)
    arguments = parser.parse_args()

    repository = SQLiteHistoricalRepository(arguments.database)
    result = MLResearchPipeline(
        repository=repository,
        calendar=load_nse_calendar(arguments.calendar_json),
    ).run(
        dataset_id=arguments.dataset_id,
        instrument_id="NIFTY50_SPOT",
        config=WalkForwardConfig(
            n_splits=arguments.splits,
            minimum_train_samples=arguments.minimum_train_samples,
            test_samples=arguments.test_samples,
            purge_bars=arguments.purge_bars,
            embargo_bars=arguments.embargo_bars,
            minimum_train_class_samples=arguments.minimum_train_class_samples,
        ),
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / f"{result.report.run_id}.json"
    payload = json.dumps(result.report.to_contract(), indent=2, sort_keys=True) + "\n"
    try:
        with output_path.open("x", encoding="utf-8") as file:
            file.write(payload)
    except FileExistsError:
        raise SystemExit(
            f"Refusing to overwrite immutable research report: {output_path}"
        ) from None

    print(
        json.dumps(
            {
                "run_id": result.report.run_id,
                "persisted": result.persisted,
                "report_path": str(output_path),
                "eligible_samples": result.report.dataset_report.eligible_samples,
                "outcome_support": dict(result.report.dataset_report.outcome_support),
                "ambiguous_labels": result.report.dataset_report.ambiguous_labels,
                "selected_research_candidate": result.report.selected_research_candidate,
                "approved_for_live_inference": False,
                "calibrated": False,
                "official_signal_available": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

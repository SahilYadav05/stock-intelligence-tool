"""Run Step 16 locked shadow research and a one-minute simulated-live backtest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from nifty_terminal.research.step16 import (
    LOCKED_ATR_MULTIPLIER,
    LOCKED_CANDIDATE,
    STEP16_VERSION,
    run_locked_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/research/locked-shadow"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/models/shadow"),
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

    print("Step 16 locked shadow research and simulated-live backtest")
    print(f"- dataset: {dataset_id}")
    print(f"- sourced exchange calendar: {arguments.calendar_json}")
    print("- locked target: symmetric 1.5 ATR first-touch over 60 minutes")
    print(f"- locked candidate: {LOCKED_CANDIDATE}")
    print("- calibration comparison: identity, temperature, prior shrinkage, vector scaling")
    print("- signal replay: next-minute entry; 1m stop/target resolution; no overlap")
    print("- conservative costs: 0.5 NIFTY points one-way slippage")
    print("- shadow artifact only; live signals and automatic trading remain disabled")
    print("- local CPU runtime may be several minutes")

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
    payload = run_locked_research(
        dataset=dataset,
        minute_candles=candles[Timeframe.M1],
    )
    created_at = datetime.now(timezone.utc)
    experiment_id = str(
        uuid5(
            NAMESPACE_URL,
            f"step16:{dataset_id}:{STEP16_VERSION}:{created_at.isoformat()}",
        )
    )
    artifact = dict(payload.pop("shadow_artifact"))
    expected_checksum = str(artifact.pop("sha256"))
    actual_checksum = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_checksum != expected_checksum:
        raise SystemExit("Shadow artifact checksum verification failed before persistence")
    artifact["sha256"] = expected_checksum

    arguments.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = arguments.artifact_dir / f"{expected_checksum}.json"
    _write_content_addressed_json(artifact_path, artifact)
    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    payload["shadow_artifact"] = {
        "path": str(artifact_path),
        "sha256": expected_checksum,
        "artifact_version": artifact["artifact_version"],
        "shadow_only": True,
        "approved_for_live_inference": False,
        "safe_json_parameters_only": True,
    }
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_dir / f"{experiment_id}.json"
    _write_immutable_json(report_path, payload)

    backtest = dict(payload["signal_backtest"])
    backtest.pop("trades", None)
    printable = {
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "locked_specification": payload["locked_specification"],
        "selected_calibration_method": payload["selected_calibration_method"],
        "historical_backtest_probability_metrics": payload[
            "historical_backtest_probability_metrics"
        ],
        "signal_backtest": backtest,
        "shadow_artifact": payload["shadow_artifact"],
        "forward_confirmation_required": True,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "report_path": str(report_path),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    print("RESULT: STEP 16 LOCKED SHADOW RESEARCH COMPLETED")
    print("No model was released for official live inference or order execution.")


def _write_content_addressed_json(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise SystemExit(
                f"Existing content-addressed artifact does not match its digest: {path}"
            )
        return
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(serialized)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise SystemExit(
                f"Concurrent artifact write produced different content: {path}"
            ) from None


if __name__ == "__main__":
    main()

"""Run Step 17 policy research and create a fail-closed shadow runtime manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.cli.run_locked_shadow_research import _write_content_addressed_json
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
from nifty_terminal.research.step17 import STEP17_VERSION, run_policy_research


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--step16-report", type=Path)
    parser.add_argument(
        "--step16-report-dir",
        type=Path,
        default=Path("artifacts/research/locked-shadow"),
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("artifacts/research/shadow-policy")
    )
    parser.add_argument(
        "--policy-dir", type=Path, default=Path("artifacts/models/shadow-policy")
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("artifacts/shadow-runtime")
    )
    arguments = parser.parse_args()

    _require_live_signal_kill_switch(arguments.env_file)
    step16_path = arguments.step16_report or _latest_json(arguments.step16_report_dir)
    step16 = _read_json(step16_path)
    if step16.get("approved_for_live_inference") is not False:
        raise SystemExit("Step 16 report is not fail-closed")
    if step16.get("locked_specification", {}).get("atr_multiplier") != "1.5":
        raise SystemExit("Step 17 requires the locked 1.5 ATR Step 16 report")
    dataset_id = arguments.dataset_id or str(step16["dataset_id"])
    if dataset_id != step16["dataset_id"]:
        raise SystemExit("Requested dataset does not match the Step 16 report")
    model_path = Path(str(step16["shadow_artifact"]["path"]))
    model_artifact = _read_json(model_path)
    _verify_embedded_checksum(model_artifact, expected=str(step16["shadow_artifact"]["sha256"]))

    repository = SQLiteHistoricalRepository(arguments.database)
    if repository.load_dataset_quality_status(dataset_id=dataset_id) is None:
        raise SystemExit(f"Historical dataset is not present: {dataset_id}")
    candles = {
        timeframe: repository.load_latest_candles(
            dataset_id=dataset_id,
            instrument_id="NIFTY50_SPOT",
            timeframe=timeframe,
        )
        for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1)
    }
    if any(not rows for rows in candles.values()):
        raise SystemExit("The locked dataset is missing a required canonical timeframe")

    print("Step 17 signal-policy research and shadow-runtime preparation")
    print(f"- dataset: {dataset_id}")
    print(f"- Step 16 report: {step16_path}")
    print(f"- shadow model SHA-256: {model_artifact['sha256']}")
    print("- policy selection: chronological fold 3 only")
    print("- policy evaluation: later chronological fold 4 only")
    print("- score sources: raw directional score and calibrated probability")
    print("- execution: next 1m open; 0.5 point one-way slippage; no overlap")
    print("- no candidate is forced to pass; WAIT-only shadow mode is valid")
    print("- official signals and automatic trading remain disabled")

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
    payload = run_policy_research(
        dataset=dataset,
        minute_candles=candles[Timeframe.M1],
        calibration_method=str(step16["selected_calibration_method"]),
    )
    created_at = datetime.now(timezone.utc)
    experiment_id = str(
        uuid5(
            NAMESPACE_URL,
            f"step17:{dataset_id}:{STEP17_VERSION}:{created_at.isoformat()}",
        )
    )
    policy_artifact = dict(payload.pop("policy_artifact"))
    _verify_embedded_checksum(policy_artifact)
    policy_path = arguments.policy_dir / f"{policy_artifact['sha256']}.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    _write_content_addressed_json(policy_path, policy_artifact)

    manifest = {
        "schema_version": 1,
        "manifest_version": "nifty_shadow_runtime.v1",
        "dataset_id": dataset_id,
        "model_artifact_path": str(model_path),
        "model_artifact_sha256": str(model_artifact["sha256"]),
        "policy_artifact_path": str(policy_path),
        "policy_artifact_sha256": str(policy_artifact["sha256"]),
        "prediction_collection_enabled": True,
        "shadow_candidate_directions_enabled": bool(
            policy_artifact["shadow_candidate_directions_enabled"]
        ),
        "runtime_mode": str(policy_artifact["runtime_mode"]),
        "official_signal_available": False,
        "precise_probability_display_allowed": False,
        "automatic_trading_enabled": False,
        "live_signal_kill_switch_required": True,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest, "sha256": manifest_hash}
    manifest_path = arguments.manifest_dir / f"{manifest_hash}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_content_addressed_json(manifest_path, manifest)

    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    payload["step16_report_path"] = str(step16_path)
    payload["shadow_model"] = {
        "path": str(model_path),
        "sha256": str(model_artifact["sha256"]),
    }
    payload["policy_artifact"] = {
        "path": str(policy_path),
        "sha256": str(policy_artifact["sha256"]),
        "runtime_mode": policy_artifact["runtime_mode"],
        "shadow_candidate_directions_enabled": policy_artifact[
            "shadow_candidate_directions_enabled"
        ],
    }
    payload["shadow_runtime_manifest"] = {
        "path": str(manifest_path),
        "sha256": manifest_hash,
    }
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_dir / f"{experiment_id}.json"
    _write_immutable_json(report_path, payload)

    evaluation = dict(payload["historical_evaluation"])
    evaluation.pop("trades", None)
    printable = {
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "candidate_count": payload["candidate_count"],
        "passing_selection_candidate_count": payload[
            "passing_selection_candidate_count"
        ],
        "selected_policy": payload["selected_policy"],
        "selection_metrics": payload["selection_metrics"],
        "selection_blockers": payload["selection_blockers"],
        "historical_evaluation": evaluation,
        "historical_evaluation_blockers": payload[
            "historical_evaluation_blockers"
        ],
        "historical_policy_gate_passed": payload["historical_policy_gate_passed"],
        "policy_artifact": payload["policy_artifact"],
        "shadow_runtime_manifest": payload["shadow_runtime_manifest"],
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "report_path": str(report_path),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    print("RESULT: STEP 17 POLICY RESEARCH COMPLETED")
    print("The runtime manifest is shadow-only; official signals remain disabled.")


def _latest_json(directory: Path) -> Path:
    paths = tuple(directory.glob("*.json"))
    if not paths:
        raise SystemExit(f"No Step 16 JSON report exists in: {directory}")
    return max(paths, key=lambda item: item.stat().st_mtime_ns)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Required JSON file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _verify_embedded_checksum(
    payload: dict[str, object], *, expected: str | None = None
) -> None:
    embedded = str(payload.get("sha256", ""))
    if expected is not None and embedded != expected:
        raise SystemExit("Artifact checksum does not match its parent report")
    body = dict(payload)
    body.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if embedded != actual:
        raise SystemExit("Artifact SHA-256 verification failed")


if __name__ == "__main__":
    main()

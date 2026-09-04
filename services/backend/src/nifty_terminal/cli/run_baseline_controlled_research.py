"""Run Step 18F baseline-controlled direction-specific research."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.cli.run_cross_market_research import _latest_context_bundle
from nifty_terminal.cli.run_real_data_research import (
    DEFAULT_CALENDAR,
    _require_live_signal_kill_switch,
    _write_immutable_json,
)
from nifty_terminal.context.bundle import bundle_sha256, read_bundle
from nifty_terminal.context.features import build_context_feature_matrix
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.labels import symmetric_first_touch_config
from nifty_terminal.research.step16 import LOCKED_ATR_MULTIPLIER
from nifty_terminal.research.step18f import (
    ARCHITECTURES,
    BENCHMARKS,
    MODEL_CANDIDATES,
    RESEARCH_IDENTITY,
    run_baseline_controlled_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--context-bundle", type=Path)
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("artifacts/research/baseline-controlled-v1"),
    )
    args = parser.parse_args()
    _require_live_signal_kill_switch(args.env_file)
    repository = SQLiteHistoricalRepository(args.database)
    dataset_id = args.dataset_id or repository.latest_pass_dataset_id(
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
        raise SystemExit("The NIFTY dataset is missing a required canonical timeframe")
    context_path = args.context_bundle or _latest_context_bundle(
        Path("artifacts/context/angelone")
    )
    bundle = read_bundle(context_path)
    print("Step 18F baseline-controlled direction-specific research")
    print(f"- NIFTY dataset: {dataset_id}")
    print(f"- context bundle: {context_path}")
    print("- objective: incremental LONG/SHORT R above causal time/regime baselines")
    print("- architectures: " + ", ".join(ARCHITECTURES))
    print("- candidates: " + ", ".join(MODEL_CANDIDATES))
    print("- hard stability gate: improvement and rank sign must hold in every selection fold")
    print("- direction-specific policy: LONG or SHORT may be independently disabled")
    print("- locked economic baselines: " + ", ".join(BENCHMARKS))
    print("- all baselines use the same costs, stop-first rule and no-overlap replay")
    print("- thresholds are frozen before the reused historical diagnostic fold")
    print("- previous history remains ineligible as fresh forward evidence")
    print("- official signals and automatic trading remain disabled")
    print("- local CPU runtime may be 60-180 minutes")
    dataset = TrainingDatasetAssembler(
        load_nse_calendar(args.calendar_json),
        label_config=symmetric_first_touch_config(LOCKED_ATR_MULTIPLIER),
    ).assemble(
        dataset_id=dataset_id,
        minute_candles=candles[Timeframe.M1],
        primary_candles=candles[Timeframe.M5],
        context_15m_candles=candles[Timeframe.M15],
        context_1h_candles=candles[Timeframe.H1],
    )
    context = build_context_feature_matrix(
        dataset=dataset,
        primary_candles=candles[Timeframe.M5],
        bundle=bundle,
    )
    payload = run_baseline_controlled_research(
        context=context,
        minute_candles=candles[Timeframe.M1],
        context_bundle_sha256=bundle_sha256(bundle),
    )
    created_at = datetime.now(timezone.utc)
    experiment_id = str(uuid5(
        NAMESPACE_URL,
        f"step18f:{dataset_id}:{RESEARCH_IDENTITY}:{created_at.isoformat()}",
    ))
    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{experiment_id}.json"
    payload["report_path"] = str(report_path)
    _write_immutable_json(report_path, payload)
    printable = {
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "context_bundle_sha256": payload["context_bundle_sha256"],
        "objective": payload["objective"],
        "selected_models": payload["selected_models"],
        "utility_diagnostics": payload["utility_diagnostics"],
        "policy_selection": payload["policy_selection"],
        "historical_simulated_live_replay": payload["historical_simulated_live_replay"],
        "research_gate": payload["research_gate"],
        "uncertainty_controls": payload["uncertainty_controls"],
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "report_path": str(report_path),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    print("RESULT: STEP 18F BASELINE-CONTROLLED RESEARCH COMPLETED")
    print("No model or signal was released; uncertainty remains explicit and fail-closed.")


if __name__ == "__main__":
    main()

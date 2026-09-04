"""Run Step 18B trade-aligned model improvement research."""

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
from nifty_terminal.research.step18b import (
    CANDIDATE_NAMES,
    RESEARCH_IDENTITY,
    STEP18B_VERSION,
    run_trade_aligned_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--report-dir", type=Path, default=Path("artifacts/research/trade-aligned-v3")
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

    print("Step 18B trade-aligned model improvement research")
    print(f"- dataset: {dataset_id}")
    print(f"- sourced exchange calendar: {arguments.calendar_json}")
    print("- outputs: long and short target-before-stop probabilities")
    print("- execution-aligned target: 1.0 ATR; stop: 0.75 ATR; horizon: 60m")
    print("- entry/replay: next 1m open; 0.5 point one-way slippage; no overlap")
    print("- features: stationary price/structure/pattern features; finalized candles only")
    print(f"- model candidates: {', '.join(CANDIDATE_NAMES)}")
    print("- baselines: historical prior and small technical logistic model")
    print("- validation: purged chronology plus session-block bootstrap confidence")
    print("- policy must independently support BUY and SELL with positive lower-bound expectancy")
    print("- no historical news or cross-market data is fabricated")
    print("- existing Step 17/18 artifacts and ledgers will not be modified")
    print("- official signals and automatic trading remain disabled")
    print("- local CPU runtime may be 20-60 minutes")

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
    payload = run_trade_aligned_research(
        dataset=dataset,
        minute_candles=candles[Timeframe.M1],
        primary_candles=candles[Timeframe.M5],
    )
    created_at = datetime.now(timezone.utc)
    experiment_id = str(
        uuid5(
            NAMESPACE_URL,
            f"step18b:{dataset_id}:{RESEARCH_IDENTITY}:{created_at.isoformat()}",
        )
    )
    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_dir / f"{experiment_id}.json"
    _write_immutable_json(report_path, payload)

    printable = {
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "target_definition": payload["target_definition"],
        "dataset": payload["dataset"],
        "feature_architecture": {
            "version": payload["feature_architecture"]["version"],
            "feature_set_hash": payload["feature_architecture"]["feature_set_hash"],
            "feature_count": payload["feature_architecture"]["feature_count"],
            "absolute_price_features_removed": True,
        },
        "selected_candidates": payload["selected_candidates"],
        "selected_calibrations": payload["selected_calibrations"],
        "probability_diagnostics": payload["probability_diagnostics"],
        "policy_selection": payload["policy_selection"],
        "historical_simulated_live_replay": payload["historical_simulated_live_replay"],
        "research_gate": payload["research_gate"],
        "known_data_limitations": payload["known_data_limitations"],
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "report_path": str(report_path),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    print("RESULT: STEP 18B TRADE-ALIGNED RESEARCH COMPLETED")
    print("No model or signal was released. A failed gate is a valid result.")


if __name__ == "__main__":
    main()

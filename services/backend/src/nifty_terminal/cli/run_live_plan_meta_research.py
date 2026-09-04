"""Run Step 26 live-plan-aligned price-action meta-label research."""

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
from nifty_terminal.features.research_v4 import build_price_action_research_matrix
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.models import QualityStatus
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.labels import symmetric_first_touch_config
from nifty_terminal.research.step16 import LOCKED_ATR_MULTIPLIER
from nifty_terminal.research.step26 import (
    RESEARCH_IDENTITY,
    STEP26_VERSION,
    run_live_plan_meta_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.sqlite3"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--context-bundle", type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/research/live-plan-meta-v1"),
    )
    args = parser.parse_args()
    _require_live_signal_kill_switch(args.env_file)
    repository = SQLiteHistoricalRepository(args.database)
    quality = repository.load_dataset_quality_status(dataset_id=args.dataset_id)
    if quality is None or quality is QualityStatus.REJECTED:
        raise SystemExit("The explicitly selected dataset is missing or rejected")
    candles = {
        timeframe: repository.load_latest_candles(
            dataset_id=args.dataset_id,
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
    print("Step 26 live-plan-aligned price-action meta-label research")
    print(f"- NIFTY dataset: {args.dataset_id}")
    print(f"- context bundle: {context_path}")
    print("- production price-action direction and exact displayed levels")
    print("- explicit T1/T2/T3 allocation, protective stops, costs and no-chase entry")
    print("- seven purged chronological folds; final two are historical diagnostics")
    print("- no artifact or official signal can be released from previously seen history")
    dataset = TrainingDatasetAssembler(
        load_nse_calendar(args.calendar_json),
        label_config=symmetric_first_touch_config(LOCKED_ATR_MULTIPLIER),
    ).assemble(
        dataset_id=args.dataset_id,
        minute_candles=candles[Timeframe.M1],
        primary_candles=candles[Timeframe.M5],
        context_15m_candles=candles[Timeframe.M15],
        context_1h_candles=candles[Timeframe.H1],
    )
    price_action = build_price_action_research_matrix(dataset, candles[Timeframe.M5])
    context = build_context_feature_matrix(
        dataset=dataset,
        primary_candles=candles[Timeframe.M5],
        bundle=bundle,
        base_matrix=price_action,
    )
    payload = run_live_plan_meta_research(
        context=context,
        minute_candles=candles[Timeframe.M1],
        primary_candles=candles[Timeframe.M5],
        context_15m_candles=candles[Timeframe.M15],
        context_1h_candles=candles[Timeframe.H1],
        context_bundle_sha256=bundle_sha256(bundle),
    )
    created_at = datetime.now(timezone.utc)
    experiment_id = str(
        uuid5(
            NAMESPACE_URL,
            f"step26:{args.dataset_id}:{RESEARCH_IDENTITY}:{created_at.isoformat()}",
        )
    )
    payload["experiment_id"] = experiment_id
    payload["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    payload["dataset_quality_status"] = quality.value
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{experiment_id}.json"
    payload["report_path"] = str(report_path)
    _write_immutable_json(report_path, payload)
    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "experiment_id": experiment_id,
                "execution_policy": payload["execution_policy_selection"]["selected"],
                "selected_model": payload["selected_model"],
                "policy_selection": payload["policy_selection"],
                "historical_diagnostic": payload["historical_diagnostic"],
                "research_gate": payload["research_gate"],
                "report_path": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

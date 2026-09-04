"""Audit the Step 20 trade-label geometry and non-model direction baselines."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from nifty_terminal.cli.run_cross_market_research import _latest_context_bundle
from nifty_terminal.context.bundle import read_bundle
from nifty_terminal.context.features import build_context_feature_matrix
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.features.research_v4 import build_price_action_research_matrix
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.labels import symmetric_first_touch_config
from nifty_terminal.research.step16 import LOCKED_ATR_MULTIPLIER
from nifty_terminal.research.step18b import build_trade_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id")
    args = parser.parse_args()
    repository = SQLiteHistoricalRepository(Path("data/research.sqlite3"))
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
    dataset = TrainingDatasetAssembler(
        load_nse_calendar(Path("config/nse-calendar-through-2026-08-25.json")),
        label_config=symmetric_first_touch_config(LOCKED_ATR_MULTIPLIER),
    ).assemble(
        dataset_id=dataset_id,
        minute_candles=candles[Timeframe.M1],
        primary_candles=candles[Timeframe.M5],
        context_15m_candles=candles[Timeframe.M15],
        context_1h_candles=candles[Timeframe.H1],
    )
    price_action = build_price_action_research_matrix(dataset, candles[Timeframe.M5])
    bundle = read_bundle(_latest_context_bundle(Path("artifacts/context/angelone")))
    context = build_context_feature_matrix(
        dataset=dataset,
        primary_candles=candles[Timeframe.M5],
        bundle=bundle,
        base_matrix=price_action,
    )
    samples, _, long_paths, short_paths, _ = build_trade_paths(
        dataset=context.dataset,
        features=context.matrix,
        minute_candles=candles[Timeframe.M1],
    )
    combinations = Counter(
        (
            long_paths[sample.sample_id].success,
            short_paths[sample.sample_id].success,
        )
        for sample in samples
    )
    better_side = Counter(
        "LONG"
        if long_paths[sample.sample_id].r_multiple
        > short_paths[sample.sample_id].r_multiple
        else "SHORT"
        if short_paths[sample.sample_id].r_multiple
        > long_paths[sample.sample_id].r_multiple
        else "TIE"
        for sample in samples
    )
    total = len(samples)
    long_wins = sum(long_paths[sample.sample_id].net_points > 0 for sample in samples)
    short_wins = sum(short_paths[sample.sample_id].net_points > 0 for sample in samples)
    print(
        {
            "dataset_id": dataset_id,
            "sample_count": total,
            "target_first_combinations": {
                f"long_{long}_short_{short}": count
                for (long, short), count in sorted(combinations.items())
            },
            "better_realized_side": dict(better_side),
            "always_long_win_rate": long_wins / total,
            "always_short_win_rate": short_wins / total,
            "balanced_random_direction_expected_win_rate": (
                long_wins + short_wins
            )
            / (2 * total),
            "unambiguous_direction_rows": combinations[(1, 0)]
            + combinations[(0, 1)],
        }
    )


if __name__ == "__main__":
    main()

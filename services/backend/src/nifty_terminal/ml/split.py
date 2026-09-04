"""Purged and embargoed expanding-window chronological validation."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

from nifty_terminal.ml.definitions import CLASS_ORDER, HORIZON_BARS, PRIMARY_TIMEFRAME_MINUTES
from nifty_terminal.ml.models import TrainingSample, WalkForwardConfig, WalkForwardFold


class PurgedWalkForwardSplitter:
    def split(
        self,
        samples: tuple[TrainingSample, ...],
        config: WalkForwardConfig,
    ) -> tuple[WalkForwardFold, ...]:
        if tuple(item.decision_time for item in samples) != tuple(
            sorted(item.decision_time for item in samples)
        ):
            raise ValueError("Training samples must be chronological")
        if len({item.decision_time for item in samples}) != len(samples):
            raise ValueError("Training decisions must have unique timestamps")
        if config.purge_bars < HORIZON_BARS:
            raise ValueError("Purge cannot be shorter than the 12-bar label horizon")

        required_test = config.n_splits * config.test_samples
        first_test_index = len(samples) - required_test
        if first_test_index < config.minimum_train_samples:
            raise ValueError(
                "Insufficient samples for the requested chronological folds: "
                f"have {len(samples)}, need at least "
                f"{config.minimum_train_samples + required_test} before purge/embargo"
            )

        extra_gap_bars = config.embargo_bars + config.purge_bars - HORIZON_BARS
        extra_gap = timedelta(minutes=extra_gap_bars * PRIMARY_TIMEFRAME_MINUTES)
        folds: list[WalkForwardFold] = []
        for fold_index in range(config.n_splits):
            test_start_index = first_test_index + fold_index * config.test_samples
            test_indices = tuple(
                range(test_start_index, test_start_index + config.test_samples)
            )
            test_starts_at = samples[test_indices[0]].decision_time
            embargo_cutoff = test_starts_at - extra_gap
            train_indices = tuple(
                index
                for index in range(test_start_index)
                if samples[index].label_window_end <= embargo_cutoff
            )
            if len(train_indices) < config.minimum_train_samples:
                raise ValueError(
                    f"Fold {fold_index} has {len(train_indices)} training samples after "
                    "purge/embargo, below the configured minimum"
                )
            support = Counter(samples[index].outcome.value for index in train_indices)
            weak = [
                name
                for name in CLASS_ORDER
                if support[name] < config.minimum_train_class_samples
            ]
            if weak:
                raise ValueError(
                    f"Fold {fold_index} has insufficient training support for: "
                    + ", ".join(weak)
                )
            maximum_label_end = max(samples[index].label_window_end for index in train_indices)
            if maximum_label_end > embargo_cutoff:
                raise AssertionError("Purged fold leaked an outcome window across its cutoff")
            folds.append(
                WalkForwardFold(
                    fold_index=fold_index,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_starts_at=samples[train_indices[0]].decision_time,
                    train_ends_at=samples[train_indices[-1]].decision_time,
                    maximum_train_label_end=maximum_label_end,
                    test_starts_at=test_starts_at,
                    test_ends_at=samples[test_indices[-1]].decision_time,
                    embargo_cutoff=embargo_cutoff,
                    train_class_support=tuple((name, support[name]) for name in CLASS_ORDER),
                )
            )
        return tuple(folds)

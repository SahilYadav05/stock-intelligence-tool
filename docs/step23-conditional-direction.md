# Step 23 — conditional opportunity and direction research

The Step 20 pooled classifier weakly ranked target-first outcomes, but it mixed two
different questions in one score: whether the next hour offers a tradeable move and
which side wins. On the pinned expanded dataset, 33,291 of 39,555 complete paths had
exactly one successful direction and 6,264 had neither. Both directions could not
succeed under the locked barrier geometry.

Step 23 therefore uses two causal heads:

- an opportunity head estimates whether exactly one side will succeed;
- a direction head, trained only on opportunity rows, estimates whether LONG is
  that successful side.

Their product creates mutually exclusive LONG and SHORT scores. Three predeclared
model candidates, two monotonic calibration choices per head, and nine fixed policy
candidates use the same seven purged chronological folds as Step 20. Dataset ID is
required by the CLI so a later dataset cannot silently replace the intended
research history. The report records the selected dataset's stored quality status;
missing and rejected datasets fail closed.

The final two folds remain historical diagnostics, all thresholds are selected
earlier, positions cannot overlap, and frequency, expectancy, profit factor,
drawdown, direction balance, and baseline uplift remain hard gates. This experiment
cannot authorize live inference; a passing historical result would still require a
frozen forward confirmation period.

## Completed expanded-data result

Experiment `3d222ba2-8530-5924-bd3f-1ef38566ec83` used the pinned Step 20 dataset.
The conditional formulation did not solve direction selection. Diagnostic direction
AUC was 0.5047 with negative Brier skill. The best rejected policy produced 728
trades across 67 sessions, a 42.99% win rate, 0.90 profit factor, -0.068R average,
and 72.99R maximum drawdown. No model or policy passed, and no artifact or official
signal was released. This result strengthens the case for genuinely new derivatives
information rather than more price-only tuning.

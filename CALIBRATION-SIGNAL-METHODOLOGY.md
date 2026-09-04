# Step 7 Calibration, Signal, and Risk Methodology

Step 7 converts Step 6 out-of-sample probability estimates into auditable
calibration research and deterministic BUY/SELL/WAIT decisions. It does not
approve a model automatically, connect licensed data, trade, or claim profit.

## Chronological calibration boundary

Only the selected candidate's immutable chronological OOS predictions are
eligible. They are sorted by decision timestamp and divided once:

- earlier 60%: fit the calibrator
- later 40%: untouched calibration evaluation

The two partitions cannot overlap. Multiclass temperature scaling fits one
positive temperature by minimizing log loss on the earlier block. It transforms
the logarithm of each raw class probability and normalizes the three values.
The artifact is plain JSON numeric metadata, not an executable pickle.

Isotonic calibration is intentionally deferred. It is more flexible but can
overfit when each class and confidence region lacks substantial support.

## Probability release gate

Precise percentages are withheld unless every gate passes:

1. At least 500 OOS predictions in total.
2. Every class has at least 50 fit and 30 untouched evaluation examples.
3. Untouched-evaluation ECE is at most 0.05.
4. Multiclass Brier skill is positive against the earlier-block class prior.
5. Calibrated Brier score and log loss do not degrade versus raw probabilities.
6. At least two confidence bins contain 30 untouched examples each.
7. Every adequately sized chronological fold slice has ECE at most 0.10 and
   Brier skill no worse than -0.10.

These are release minima, not evidence that a strategy is profitable. The
dashboard can display a precise probability only when the calibration release
passes and the current probability falls in a supported bin.

## Deterministic WAIT-first policy

The policy is versioned as `wait_first_atr_policy.v1`. It cannot call an LLM or
change model probabilities. Any hard-gate failure returns WAIT without entry,
stop, or target levels. Hard gates include:

- data status is not LIVE
- chart/model snapshot revision mismatch
- developing rather than finalized primary candle
- missing finalized 15-minute or 1-hour context
- unavailable feature snapshot or ATR
- failed calibration release or unsupported confidence bin
- active event-risk gate
- probability, class-margin, expected-value, or reward/risk failure
- conflict with an active opposite signal

The initial conservative research defaults are 0.60 directional probability,
0.15 margin over the strongest alternative, maximum NEITHER probability 0.45,
and minimum expected value 0.15 ATR. These values were not fixed by Phase 1 or
Phase 2 and are explicitly marked as research defaults. They must later be
validated without tuning on the final test set.

## Underlying-only risk levels

Levels apply to NIFTY 50 spot, never an option strike:

- entry zone: finalized close ±0.10 ATR
- stop and invalidation: 0.75 ATR against the direction
- target 1: 1.00 ATR
- target 2: 1.50 ATR
- target 3: 2.00 ATR
- minimum target-1 reward/risk: 1.25
- expiry: 12 finalized 5-minute bars

BUY and SELL calculations are symmetric. No levels are emitted for WAIT.

## Hysteresis and lifecycle

An active BUY cannot become SELL, or vice versa, in one decision. An opposite
candidate must satisfy the stronger 0.72 probability and 0.25 margin tests, and
the prior active signal must still be invalidated or expire before replacement.

Original decisions are immutable. TARGET HIT, STOP HIT, INVALIDATED, EXPIRED,
MAINTAINED, UPGRADED, and DOWNGRADED are stored as separate lifecycle events.
If one candle touches stop and target and no lower-resolution order is known,
the assessment is INVALIDATED as ambiguous rather than guessed.

## Current release state

The repository contains no real approved model or calibration artifact and no
licensed live provider configuration. Therefore the terminal truthfully shows
`LIVE ANALYSIS UNAVAILABLE`, `NO APPROVED CALIBRATION`, and `NO SIGNAL`.

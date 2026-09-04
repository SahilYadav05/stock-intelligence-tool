# Step 18F — Baseline-controlled direction-specific research

Step 18E produced a mildly positive historical replay, but the complete report
showed that the selected SHORT model's rank correlation was negative on both
model-selection folds and positive only on a later, already examined period.
LONG scores ranked outcomes but every diagnostic decile still had negative raw
expectancy. The replay could therefore reflect broad directional period bias,
not a repeatable model edge.

Step 18F corrects that ambiguity without lowering a gate.

## Modelling correction

- A train-only baseline estimates expected LONG and SHORT utility from shrunk
  time-of-day, trend and ADX regimes.
- Each candidate learns the residual utility above that baseline rather than the
  raw directional outcome.
- Training observations receive session-balanced weights.
- Ridge, squared-loss boosting, robust absolute-loss boosting, regularized Extra
  Trees and a median ensemble are compared across NIFTY-only, NIFTY+Bank Nifty,
  NIFTY+VIX and all-context feature sets.
- A direction is ineligible if MSE improvement, incremental rank sign, or the
  top-cohort excess utility fails on either model-selection fold.

## Economic correction

LONG and SHORT may be enabled independently. A frozen policy must establish raw
positive expectancy and positive excess expectancy above the causal regime
baseline for every enabled direction. It must also beat all five locked
comparators using session-block confidence intervals:

1. WAIT
2. Always LONG
3. Always SHORT
4. Technical trend/ADX
5. Causal time/regime direction

Every comparator uses the same next-minute entry, conservative stop-first path,
slippage and no-overlap execution rules.

## Remaining uncertainty

Uncertainty is controlled and disclosed; it is not claimed to be eliminated.
All existing historical periods have influenced model development. Even a clean
Step 18F historical result remains ineligible for release until a frozen model
passes genuinely future confirmation. Missing legitimate historical futures
volume/OI, point-in-time breadth and timestamped news are still explicit data
limitations.

Step 18F never creates a deployable model, enables precise probabilities,
modifies the existing shadow runtime, emits an official signal, or enables
automatic trading.

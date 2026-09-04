# Step 19 — causal price action and market structure

Step 19 adds information that the prior baseline-controlled experiment did not
explicitly encode: confirmed swing progression, breaks of structure, liquidity
sweeps, short-term range compression, close location and candle efficiency.

Every pivot needs two finalized candles on both sides. A pivot at time `t` is
therefore unavailable until `t+2`; future-confirmed pivots and developing candles
never enter a decision row. Values are normalized by decision-time ATR.

The Step 18F chronology, candidates, costs, stop-first resolution, no-overlap
execution and hard gates remain unchanged. This isolates whether the added price
action information improves stability rather than manufacturing a pass by
changing thresholds.

The live terminal separately exposes a deterministic, conditional price-action
plan with a trigger, structure-aware invalidation, stop and targets at 1.25R,
2R and 3R. It is research-only, shows no probability and cannot place an order.

Even if the historical Step 19 gate passes, the model remains ineligible for an
official signal until a frozen artifact succeeds on genuinely future sessions.

## Completed historical experiment

Run `4f1c561f-0aed-54e9-8baf-0f10d93c1732` used canonical NIFTY data set
`5d729e85-e784-5c47-b04b-cd0ece3cb9a5` and 2,000 walk-forward evaluation
decisions. No policy passed the unchanged selection gate, so the result does
not release a model or an official signal.

The strongest rejected exploratory policy was short-only at a 0.80 threshold:
59 paper-simulated trades, 59.3% win rate, 1.64 profit factor and a 0.13R lower
95% confidence bound on excess R. That remains insufficient evidence: it had
only 22 sessions and 59 trades, and its daily uplift lower bound was not
positive against the always-short and technical-trend baselines. The selected
long and short candidates also failed model-selection stability checks. These
results are retained as research evidence, not turned into a discretionary
override or a claim of deployable trading performance.

# Step 9 — Tracking and Monitoring Contract

Step 9 observes the research system; it does not execute trades.

## Immutable event order

1. Register the original prediction, probability vector, model/calibration
   versions, snapshot ID, and candle-revision checksum.
2. If and only if the deterministic policy produces an active BUY or SELL with
   valid levels, create one paper plan. WAIT creates no paper position.
3. Assess later finalized candles. Entry, exit, invalidation, and expiry are new
   events; the original plan and prediction are never edited.
4. After the full 60-minute label window, append a separate UP/DOWN/NEITHER
   assessment using the same first-touch rules as training and replay.

## Conservative paper rules

- Unit: NIFTY underlying index points, normalized to one unit.
- No orders, options, brokerage account, cash P&L, or return claim.
- BUY uses the high edge of the entry zone and SELL uses the low edge as a
  conservative simulated fill.
- A candle that could contain both entry and exit, or both stop and target,
  cannot reveal intrabar order and is marked INVALIDATED.
- Developing candles never create official paper lifecycle events.

## Analytics gates

Counts and coverage may be shown immediately. Accuracy, multiclass Brier score,
expected calibration error, paper win rate, and total paper points remain null
until at least 30 eligible assessed observations exist in that metric family.
Every view carries its sample count and blocker.

These metrics are diagnostics, not promises of future performance.

## Monitoring

The monitoring view reports market-data state, freshness, exact chart/model
snapshot synchronization, approved model/calibration availability, outcome
tracking coverage, and drift-readiness. Missing evidence is UNAVAILABLE; unsafe
live state is CRITICAL. External alert delivery is intentionally disabled until
Step 10 security and deployment hardening.

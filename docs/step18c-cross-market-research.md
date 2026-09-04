# Step 18C — Canonical cross-market context and controlled model comparison

Step 18B showed that NIFTY-only price features did not produce a reliable tradeable edge. Step 18C adds new information instead of repeatedly tuning indicators on the same data.

## Added now

- Angel One finalized 1-minute Bank Nifty index history.
- Angel One finalized 1-minute India VIX history.
- Immutable gzip bundle with a SHA-256 content identity.
- Exact finalized five-minute point-in-time joins; missing minutes are never filled.
- Stationary return, volatility, trend, shock and cross-market interaction features.
- A controlled NIFTY-only versus context-aware comparison using the same samples, folds, labels, execution assumptions and candidate learners.
- Incremental-value gates: both LONG and SHORT must improve AUC and Brier skill, proper-score confidence bounds must be positive, and the replay must support both directions with positive lower-bound expectancy.

## Deliberately not claimed

- A model is not released by this step.
- Historical news is not fabricated from today's headlines.
- NIFTY spot volume and VWAP remain unavailable.
- Expired futures cannot be reconstructed from the current contract master; continuous futures/OI remains blocked until a legitimate history source is added.
- Constituent breadth is not used without point-in-time membership and corporate-action-correct history.

The correct outcome may still be a failed gate. That result means more informative licensed history or a different target is required; thresholds must not be weakened to manufacture approval.

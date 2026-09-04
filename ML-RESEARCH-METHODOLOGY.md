# Step 6 ML Research Methodology

This implementation is research-only. It trains and evaluates models but does
not approve a model for live inference, calibrate a probability, create a
BUY/SELL/WAIT policy, or report trading performance.

## Locked target

- Instrument: `NIFTY50_SPOT`
- Decision input: latest finalized 5-minute candle
- Reference price: that candle's close
- Volatility: Wilder ATR(14) calculated from finalized 5-minute candles only
- Horizon: the next 12 finalized 5-minute candles, exactly 60 minutes
- UP: `close + 1.0 ATR` is touched first
- DOWN: `close - 1.0 ATR` is touched first
- NEITHER: neither barrier is touched within the horizon
- AMBIGUOUS: both barriers are touched before stored lower-resolution candles
  can establish their order; excluded from training and reported

The entire outcome window must fit inside the same explicit NSE session.
Incomplete windows, missing candles, feature blockers, and unresolved double
touches are never silently converted into NEITHER.

## Point-in-time inputs

The training row uses the finalized 5-minute feature row plus the most recent
finalized 15-minute and 1-hour feature rows whose close timestamps are no later
than the 5-minute decision timestamp. The same Step 5 causal feature function
is used. Developing candles and future higher-timeframe candles cannot enter a
training row.

## Validation

Validation is expanding-window chronological walk-forward only. Random train
or test shuffling is not available. Each fold removes training observations
whose 60-minute label windows could overlap the test boundary, then applies an
additional embargo. Every fold must retain all three outcome classes with the
configured minimum support.

The fixed Step 6 candidates are:

1. Class-frequency prior baseline
2. Balanced multinomial logistic regression
3. Balanced histogram gradient boosting challenger

There is no automated hyperparameter search in Step 6. Candidate comparison is
based first on out-of-sample multiclass Brier score, then log loss and balanced
accuracy. Accuracy, class support, class recall, raw 10-bin ECE, and fit and
inference latency are also retained. Raw ECE is diagnostic only; calibration is
Step 7.

## Historical simulated-live replay

Every out-of-sample fold prediction is created as if it were generated at its
historical decision timestamp. It stores the exact input revision checksum,
fold model identity, and raw uncalibrated class probabilities. It contains no
actual outcome and no signal.

When the 60-minute window completes, a separate immutable assessment record is
created. Original predictions are never updated with hindsight.

## Persistence and release boundary

SQLite retains append-only label, run, fold, prediction, and assessment tables.
Executable deployment-model packaging is intentionally deferred until Step 7,
where a candidate must be calibrated and pass explicit release gates. Raw Step
6 probabilities must never be shown on the trading dashboard as confidence.

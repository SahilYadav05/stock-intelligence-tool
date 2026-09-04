# Step 14 — Real-data model research and chronological validation

This step trains the existing price-only MVP candidates on the latest immutable
PASS Angel One dataset, creates purged chronological out-of-sample predictions,
fits probability calibration on an earlier OOS partition, evaluates it on a
later untouched partition, and replays the WAIT-first policy.

It does not create a deployable model artifact, enable a live signal, consume a
developing candle, use NIFTY spot volume/VWAP, ingest news or place an order.

## 1. Stop the backend

Press `Ctrl+C` in the backend Command Prompt. The frontend may remain open.

## 2. Back up the files being changed

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step14-backup.zip package.json scripts services\backend\src\nifty_terminal services\backend\tests
```

Never put `.env`, databases, artifacts or Angel One credentials in a shared
archive.

## 3. Extract the update

Download `step14-real-data-research.zip` to Downloads and run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step14-real-data-research.zip" -C .
```

## 4. Validate the complete project

```cmd
npm.cmd run check:step14
npm.cmd run test:windows
```

Do not continue unless both commands pass.

## 5. Confirm the research safety switch

Open `.env`:

```cmd
notepad .env
```

Confirm this remains exactly:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

Do not change provider credentials and do not share this file.

## 6. Run the first real-data experiment

```cmd
npm.cmd run research:real-data:windows
```

The command automatically selects the newest NIFTY50_SPOT dataset whose
immutable quality status is PASS. It runs five expanding chronological folds
with 2,000 later observations per fold. Training can take several minutes on a
local CPU.

The result can legitimately fail. A failed gate is evidence, not an application
error. Do not weaken thresholds or add indicators until the failure is
understood.

Share only these output sections:

```text
dataset_id
run_id
calibration_id
dataset
prior_baseline_metrics
selected_candidate
selected_candidate_metrics
calibration
research_gate
report_path
```

Never share `.env`, the SQLite database, provider credentials, raw licensed
candles, JWT/feed tokens or TOTP values.

## What the current model actually consumes

- Finalized 5-minute OHLC price features: returns, ranges, candle body/wicks,
  SMA/EMA trend, ATR, RSI, rolling volatility, Bollinger position, rate of
  change, breakouts and session timing.
- Only finalized 15-minute and 1-hour versions of the same causal feature set.
- No developing candle.
- No NIFTY spot volume and therefore no fake VWAP.
- No news in Step 14.

Adding every named candlestick/chart pattern is not a reliable shortcut. Many
are redundant encodings of OHLC geometry and can increase overfitting. New
features will be versioned and admitted only when they improve later untouched
data rather than the training sample.

## Completion gate

- The Step 14 structural and full regression tests pass.
- One immutable real-data experiment report exists.
- All predictions used for evaluation are chronological and out of sample.
- Calibration fit timestamps end before its evaluation timestamps begin.
- The report truthfully records PASS or blockers.
- `approved_for_live_inference` remains false.
- `LIVE_SIGNAL_KILL_SWITCH` remains true.
- No official BUY/SELL signal or order is created.

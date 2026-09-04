# Step 13 — Long-range Angel One historical research dataset

Step 13 acquires finalized NIFTY 50 one-minute history from Angel One, builds
canonical 5m/15m/1h candles with the same candle engine used by live mode,
persists immutable revisions to local SQLite, materializes the existing causal
feature set, constructs the existing 60-minute first-touch labels, and reports
whether there is enough clean class support to begin model research.

It does not train, calibrate, approve, or deploy a live model. It never places
orders, and `LIVE_SIGNAL_KILL_SWITCH=true` remains mandatory.

## 1. Stop the frontend and backend

Press `Ctrl+C` in both development-server Command Prompt windows. The history
command opens its own Angel One session, so the normal backend must not be using
the same API credentials at the same time.

## 2. Open the project

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

## 3. Create a recovery archive

```cmd
tar -a -c -f pre-step13-backup.zip package.json config services\backend\src\nifty_terminal services\backend\tests scripts
```

Do not include `.env`, `data`, `artifacts`, logs or API credentials in archives
that you share.

## 4. Extract the Step 13 update

Download `step13-update.zip` to your Downloads folder, then run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step13-update.zip" -C .
```

The archive contains complete replacement files and does not contain `.env`.

## 5. Run structural and regression tests

```cmd
npm.cmd run check:step13
```

Expected line:

```text
Step 13 check passed (6 required paths).
```

Run all tests:

```cmd
npm.cmd run test:windows
```

## 6. Acquire the research history

Make sure the computer stays awake and the internet connection remains active,
then run:

```cmd
npm.cmd run acquire:history:angelone:windows
```

The default range is `2025-01-01` through the latest fully completed and
calendar-verified NSE session, up to `2026-08-25`. Requests are split into
seven-day chunks with rate-control delays. The command may take several minutes.

It creates these local ignored files:

```text
data\research.sqlite3
artifacts\research\history\<dataset-id>.json
```

These files may be large. They remain local and are excluded from Git.

## 7. Read the final research gate

At the end, inspect these JSON fields:

```text
quality.status
quality.missing_minutes
source_minute_rows
canonical_candle_counts
training_dataset_summary.eligible_samples
training_dataset_summary.outcome_support
training_research_ready
training_readiness_blockers
```

The required result for continuing is:

```text
"quality": {
  "status": "PASS"
}

"training_research_ready": true
```

This means only that the dataset is sufficiently complete for chronological
research. It does not mean the model will be accurate or profitable.

If the command reports `DEGRADED` or `REJECTED`, do not train. Share only:

```text
quality
canonical_candle_counts
training_readiness_blockers
```

Never share `.env`, the database, provider logs, JWTs, feed tokens, TOTP values,
MPIN, API key, client code or source rows licensed for private use.

## 8. Optional explicit date range

The bundled exchange calendar is explicitly verified from `2025-01-01` through
`2026-08-25`. A request outside that interval is intentionally blocked rather
than guessing exchange holidays.

Within that range, an explicit acquisition can be run as:

```cmd
npm.cmd run acquire:history:angelone:windows -- --from-date 2025-01-01 --to-date 2026-08-25
```

## 9. Restart the terminal after acquisition

Backend:

```cmd
npm.cmd run backend:dev:windows
```

Frontend in another Command Prompt:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
npm.cmd run dev:windows
```

The live dashboard remains in `WAIT` because Step 13 creates research data, not
an approved production artifact.

## Step 13 completion gate

Step 13 is complete only when:

- `check:step13` passes;
- the complete Windows test suite passes;
- the historical command completes without exposing credentials;
- the exact source-row SHA-256 and immutable dataset ID are recorded;
- the exchange calendar covers the entire requested range;
- NIFTY spot volume is null in every row;
- quality status is `PASS` with zero missing expected minutes;
- canonical 1m/5m/15m/1h counts are reported;
- causal features and first-touch label support are reported;
- `training_research_ready=true`;
- live inference, calibrated probabilities, signals and trading remain disabled.

# Step 13.1 — NSE special-session and historical-quality correction

The first Step 13 run downloaded the Angel One history correctly, but the
calendar omitted the NSE Budget sessions on 1 February 2025 and 1 February
2026. This hotfix adds both officially announced full trading sessions and
handles only the explicitly documented 2025 Muhurat pre-open observations as
non-continuous-trading data.

The previous rejected dataset and report remain immutable audit records. Do not
delete or rewrite them.

## 1. Stop the frontend and backend

Press `Ctrl+C` in both development-server Command Prompt windows.

## 2. Open the project and back up the affected files

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step13.1-backup.zip config\nse-calendar-through-2026-08-25.json scripts\check-step13.mjs services\backend\src\nifty_terminal\history services\backend\src\nifty_terminal\cli\acquire_angelone_history.py services\backend\tests\test_angelone_history_acquisition.py
```

## 3. Extract the hotfix

Download `step13.1-quality-hotfix.zip` to Downloads, then run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step13.1-quality-hotfix.zip" -C .
```

The archive contains no `.env`, credentials, database, provider tokens or
market-data rows.

## 4. Verify the update

```cmd
npm.cmd run check:step13
npm.cmd run test:windows
```

Expected structural result:

```text
Step 13 check passed (7 required paths).
```

## 5. Run the corrected acquisition

Keep the computer awake and leave the normal backend stopped:

```cmd
npm.cmd run acquire:history:angelone:windows
```

This re-download is intentional. The old rejected database record stores its
quality report but not rejected raw provider rows. The corrected, filtered
dataset receives a new immutable dataset ID and SHA-256.

## 6. Interpret the result

The report now distinguishes:

```text
raw_source_minute_rows
source_minute_rows
session_normalization
expected_minute_coverage
quality
research_quality_accepted
training_research_ready
training_readiness_blockers
```

Missing provider minutes are never fabricated or forward-filled. Incomplete
5-minute/15-minute/1-hour buckets are not finalized, feature windows affected
by intraday gaps are blocked, and 60-minute labels requiring missing future
candles are excluded.

A small sparse gap rate may retain `quality.status=DEGRADED` while still setting
`research_quality_accepted=true`. This requires all of the following:

- at least 99.5% expected-minute coverage;
- no more than 15 missing minutes in any session;
- no consecutive missing gap longer than 5 minutes;
- no unapproved warning or error;
- zero fabricated candles.

Continue only if:

```text
research_quality_accepted: true
training_research_ready: true
```

If either is false, send only these sections:

```text
session_normalization
expected_minute_coverage
quality
canonical_candle_counts
training_dataset_summary
training_readiness_blockers
```

Never send `.env`, API credentials, tokens, the SQLite database or licensed
source market-data rows.

Signals, calibrated probabilities, news effects and order execution remain
disabled after this hotfix.

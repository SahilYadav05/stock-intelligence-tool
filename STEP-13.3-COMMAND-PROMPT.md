# Step 13.3 — NSE Closing Auction Session awareness

NSE introduced the Closing Auction Session (CAS) in the equity cash segment on
3 August 2026. For the NIFTY MVP, ordinary continuous-session candles now end
at 15:15 IST. Indicative index observations disseminated from 15:15 through the
auction are context observations, not standard one-minute OHLC model inputs.

This update is based on NSE circulars CMTR/73362 and CMTR/75479. It does not
enable model inference, signals, news scoring or order execution.

## 1. Stop the application

Press `Ctrl+C` in the backend and frontend Command Prompt windows.

## 2. Open the project and create a recovery archive

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step13.3-backup.zip package.json config scripts services\backend\src\nifty_terminal services\backend\tests src\components\live-market-terminal.tsx
```

Do not include `.env`, `data`, `artifacts`, logs or provider credentials in any
archive you share.

## 3. Extract the update

Download `step13.3-cas-update.zip` to Downloads and run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step13.3-cas-update.zip" -C .
```

## 4. Run the structural and complete regression gates

```cmd
npm.cmd run check:step13:cas
```

Expected:

```text
Step 13.3 CAS check passed (10 required paths).
```

Then run:

```cmd
npm.cmd run test:windows
```

Do not continue if either command fails.

## 5. Verify Angel One with the CAS-aware boundary

```cmd
npm.cmd run verify:angelone:windows
```

The verifier now reports continuous finalized candles and separate
closing-auction observations. CAS observations are never sent to the candle
engine.

## 6. Build the corrected full historical dataset

Use the complete range through 25 August. Do not use the earlier 31 July
workaround:

```cmd
npm.cmd run acquire:history:angelone:windows
```

The request must download again because rejected raw provider rows were not
stored. Keep the previous dataset IDs and JSON reports as immutable audit
records.

For the observed Angel One dataset, the corrected result should be close to:

```text
raw provider observations: 152853
continuous-session rows:   152805
excluded observations:     48
expected-minute coverage:  100%
quality.status:             PASS
research_quality_accepted: true
```

The 48 exclusions should consist of 12 previously audited 2025 Muhurat
non-continuous observations plus approximately 36 CAS observations from
3–25 August 2026. The exact report remains authoritative.

Continue only if:

```text
quality.status: PASS
expected_minute_coverage.missing_minutes: 0
research_quality_accepted: true
training_research_ready: true
```

If the result differs, share only:

```text
closing_auction_policy
session_normalization
expected_minute_coverage
quality
canonical_candle_counts
training_dataset_summary
training_readiness_blockers
```

Never share `.env`, API keys, client code, PIN, TOTP secret, JWT, feed token,
SQLite database or licensed raw source rows.

## 7. Restart locally

Backend:

```cmd
npm.cmd run backend:dev:windows
```

Frontend in a second Command Prompt:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
npm.cmd run dev:windows
```

During CAS, the dashboard retains the final continuous chart and displays:

```text
CLOSING AUCTION · STANDARD SIGNAL DISABLED
```

Standard 60-minute labels and signals are permitted only when their full
outcome horizon can end by the continuous-session close. From 3 August 2026,
the last standard decision time is therefore 14:15 IST.

## Completion gate

- Date-effective continuous close is 15:30 before 3 August 2026 and 15:15 from
  that date.
- Reference, order-entry and matching phases are explicitly represented.
- CAS observations are counted and audited separately.
- No CAS observation can mutate a standard developing or finalized candle.
- Historical quality expects no continuous candles during CAS.
- The standard 60-minute label window cannot cross 15:15.
- The browser shows a CAS-specific safe state while retaining the last valid
  continuous chart.
- NIFTY spot volume remains null.
- Live model inference, BUY/SELL signals and order execution remain disabled.

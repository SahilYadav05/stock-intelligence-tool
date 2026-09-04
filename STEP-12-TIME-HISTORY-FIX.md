# Step 12 correction — IST chart time and multi-day canonical history

This correction fixes two confirmed issues:

1. Lightweight Charts was displaying UTC clock labels even though candle data
   represented NSE trading. The axis and crosshair now explicitly format time
   in `Asia/Kolkata`.
2. The WebSocket market-state payload contains only the compact model-input
   window. A version-checked chart-history endpoint now loads a larger display
   window once from the same canonical candle store.

The backend still stores timestamps in UTC for auditing. Only presentation is
converted to IST; candle instants are not shifted or rewritten.

## 1. Stop both development servers

In each Command Prompt running the frontend or backend, press:

```text
Ctrl+C
```

## 2. Open the project

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

## 3. Create a recovery archive

```cmd
tar -a -c -f pre-step12-time-history-fix.zip package.json .env.example services\backend\src\nifty_terminal services\backend\tests src\components src\lib scripts
```

Do not include `.env` because it contains private credentials.

## 4. Extract the correction

Download `step12-time-history-fix.zip` to Downloads, then run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step12-time-history-fix.zip" -C .
```

All files in the archive are complete replacement files. Your `.env` is not
included or overwritten.

## 5. Increase the existing local history bootstrap

```cmd
notepad .env
```

Replace the existing `LIVE_HISTORY_LOOKBACK_DAYS` line and add the three chart
limits so the following values exist exactly once:

```text
LIVE_HISTORY_LOOKBACK_DAYS=14
LIVE_CHART_HISTORY_PRIMARY_LIMIT=750
LIVE_CHART_HISTORY_CONTEXT_LIMIT=250
LIVE_CHART_HISTORY_HOURLY_LIMIT=120
```

Keep these existing safety settings unchanged:

```text
LIVE_MINUTE_FINALIZATION_DELAY_SECONDS=5
LIVE_SIGNAL_KILL_SWITCH=true
```

Fourteen calendar days normally provide roughly ten NSE trading sessions. The
actual count can be lower because of weekends, exchange holidays, special
sessions, or provider availability.

## 6. Run the correction gate

```cmd
npm.cmd run check:step12:time-history
```

Expected result:

```text
Step 12 time/history check passed (4 required paths).
```

Then run the complete suite:

```cmd
npm.cmd run test:windows
```

No new package installation is required.

## 7. Restart the backend

```cmd
npm.cmd run backend:dev:windows
```

The first startup may take longer because it now requests a larger finalized
1-minute history window and reconstructs canonical 5m/15m/1h candles.

## 8. Restart the website

In another Command Prompt:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
npm.cmd run dev:windows
```

Open:

```text
http://127.0.0.1:5173
```

Perform a hard refresh:

```text
Ctrl+Shift+R
```

## Expected result

- NSE chart labels display Indian Standard Time.
- A regular session begins at `09:15` and its final 5-minute candle begins at
  `15:25`, closing at `15:30` IST.
- The crosshair timestamp explicitly ends with `IST`.
- The chart initially displays approximately the latest 150 candles.
- Scroll or drag left to inspect up to 750 finalized 5-minute candles.
- 15-minute and 1-hour selections also have expanded finalized history.
- Chart-history data and live snapshots originate from the same canonical
  Angel One-backed candle store.
- The chart-history response is rejected if its decision time or candle
  revision does not match the active canonical snapshot.
- Developing candles remain visual only.
- NIFTY spot volume remains null.
- BUY/SELL remains disabled until an ML model and calibration release pass all
  validation gates.

## Important quantitative distinction

More visible chart candles do not automatically make a prediction trustworthy.
The live feature window remains deterministic and point-in-time safe. Model
training, walk-forward validation, calibration and historical analog research
must use the separate historical research dataset. This correction increases
chart context without silently changing the model or fabricating an analysis.

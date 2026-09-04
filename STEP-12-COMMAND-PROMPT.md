# Step 12 — Continuous Angel One feed, canonical snapshots, and live chart

This step connects the verified Angel One adapter to the backend lifecycle. It
continuously recovers finalized 1-minute history, deterministically builds
finalized 5m/15m/1h candles, publishes the same market-state snapshot to the
browser, and updates the 5-minute developing candle from WebSocket ticks.

It does **not** place orders or enable BUY/SELL signals. The release kill switch
remains active.

## 1. Open Command Prompt in the project

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

## 2. Create a recovery archive

```cmd
tar -a -c -f pre-step12-backup.zip package.json .env.example services\backend\src\nifty_terminal services\backend\tests src\components scripts
```

Do not include `.env` in backups or uploads because it contains credentials.

## 3. Extract the complete Step 12 replacement files

Download `step12-update.zip` to your Downloads folder, then run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step12-update.zip" -C .
```

The archive deliberately does not contain `.env`, so your Angel One secrets are
not replaced.

## 4. Add the live-runtime settings to the existing `.env`

```cmd
notepad .env
```

Keep your existing credentials and ensure these values exist exactly once:

```text
MARKET_DATA_MODE=live
MARKET_DATA_PROVIDER=angelone

LIVE_HISTORY_LOOKBACK_DAYS=7
LIVE_HISTORY_RECOVERY_MINUTES=15
LIVE_HISTORY_POLL_SECONDS=10
LIVE_MINUTE_FINALIZATION_DELAY_SECONDS=5
LIVE_TICK_FRESH_SECONDS=3
LIVE_TICK_STALE_SECONDS=15
LIVE_CHART_PUBLISH_INTERVAL_MILLISECONDS=250

LIVE_SIGNAL_KILL_SWITCH=true
```

Do not share the file or paste its secret values into chat.

## 5. Run the Step 12 gate

```cmd
npm.cmd run check:step12
```

Then run the complete verification:

```cmd
npm.cmd run test:windows
```

Expected Step 12 line:

```text
Step 12 check passed (5 required paths).
```

## 6. Start the backend

In the first Command Prompt:

```cmd
npm.cmd run backend:dev:windows
```

Leave this window open. The backend now owns the Angel One session, recovery,
candle finalization, snapshot publication, and browser WebSocket.

## 7. Inspect provider health safely

Open a second Command Prompt:

```cmd
curl.exe http://127.0.0.1:8000/api/v1/provider/health
```

This response contains status and counters only. It never returns credentials,
JWTs, refresh tokens, feed tokens, MPIN, or TOTP material.

Outside the NSE regular session, `data_status` should be `MARKET_CLOSED`. During
the regular session, it should progress through `CONNECTING`/`RECOVERING` and
then become `LIVE` after the first valid tick.

## 8. Start the website

In the second Command Prompt:

```cmd
npm.cmd run dev:windows
```

Open:

```text
http://127.0.0.1:5173
```

Expected behavior:

- finalized chart history comes from authoritative Angel One 1-minute candles;
- the 5-minute developing candle and current price update from Angel One ticks;
- 15m and 1h views contain finalized context only;
- NIFTY 50 volume stays unavailable/null;
- the browser never connects directly to Angel One;
- chart and analysis transport share one canonical market-state snapshot;
- stale, disconnected, or mismatched state suppresses analysis;
- BUY/SELL stays disabled because there is no approved live model release yet.

If the market is closed, retained finalized candles may still be viewed, but the
terminal must show `MARKET CLOSED` and cannot create a new signal.

## 9. If something fails

Safe diagnostic commands are:

```cmd
curl.exe http://127.0.0.1:8000/api/v1/health
```

```cmd
curl.exe http://127.0.0.1:8000/api/v1/provider/health
```

Share only the sanitized `reason`, `data_status`, and `last_error_type` fields.
Never share `.env`, provider logs, headers, tokens, QR codes, or screenshots that
contain credentials.

## Step 12 completion gate

Step 12 is complete only when:

- `check:step12` passes;
- all backend and frontend tests pass;
- backend startup authenticates Angel One without exposing credentials;
- provider health is reachable and contains no secrets;
- finalized 5m/15m/1h candles are built only from finalized 1m history;
- a WebSocket tick updates the visual developing 5m candle during market hours;
- the developing candle ID is absent from all model-input candle IDs;
- browser and backend display the same snapshot ID and candle revision;
- stale/disconnected/market-closed states suppress live analysis;
- NIFTY spot volume is null;
- `LIVE_SIGNAL_KILL_SWITCH=true` remains active;
- no order endpoint or automatic execution path exists.

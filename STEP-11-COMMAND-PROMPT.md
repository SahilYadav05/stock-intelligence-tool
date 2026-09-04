# Step 11 — Angel One provider adapter and safe connectivity verification

This step adds Angel One behind the existing provider-neutral boundary. It does
not enable signals, order placement, or browser access to provider credentials.

## 1. Open Command Prompt in the project

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

## 2. Create a recovery archive before replacing files

```cmd
tar -a -c -f pre-step11-backup.zip package.json .env.example .gitignore services\backend scripts
```

Do not include `.env` in archives or uploads because it will contain secrets.

## 3. Extract the Step 11 update

Download `step11-update.zip` to your Downloads folder, then run:

```cmd
tar -xf "%USERPROFILE%\Downloads\step11-update.zip" -C .
```

The archive contains complete replacement files and preserves `.env`.

## 4. Install the Angel One provider dependencies

```cmd
.venv\Scripts\python.exe -m pip install -e "services\backend[server,test,research,provider-angelone]"
```

## 5. Configure backend-only credentials

Open the existing `.env`; do not replace it with `.env.example`:

```cmd
notepad .env
```

Ensure these lines exist and fill only the four private values locally:

```text
MARKET_DATA_MODE=live
MARKET_DATA_PROVIDER=angelone

ANGELONE_API_KEY=PASTE_YOUR_API_KEY_HERE
ANGELONE_CLIENT_CODE=PASTE_YOUR_ANGEL_ONE_CLIENT_CODE_HERE
ANGELONE_PIN=PASTE_YOUR_4_DIGIT_ANGEL_ONE_MPIN_HERE
ANGELONE_TOTP_SECRET=PASTE_YOUR_BASE32_TOTP_SEED_HERE

ANGELONE_NIFTY_WEBSOCKET_TOKEN=99926000
ANGELONE_NIFTY_HISTORICAL_TOKEN=99926000
ANGELONE_WEBSOCKET_EXCHANGE_TYPE=1
ANGELONE_PRICE_SCALE=100
ANGELONE_CONNECT_TIMEOUT_SECONDS=20
ANGELONE_STREAM_QUEUE_CAPACITY=4096

LIVE_SIGNAL_KILL_SWITCH=true
```

`ANGELONE_TOTP_SECRET` is the Base32 key displayed with the SmartAPI TOTP QR
code. It is not the API app secret and it is not the current six-digit code.

Never paste `.env`, credentials, JWTs, refresh tokens, feed tokens, screenshots
containing keys, or SmartAPI log files into chat.

## 6. Run the structural and backend tests

```cmd
npm.cmd run check:step11
```

```cmd
npm.cmd run test:backend:windows
```

Then run the complete project verification:

```cmd
npm.cmd run test:windows
```

## 7. Verify Angel One using real data

```cmd
npm.cmd run verify:angelone:windows
```

The verifier performs only these operations:

- authenticates server-side using API key, client code, MPIN, and generated TOTP;
- opens the official market-data WebSocket;
- subscribes to NIFTY 50;
- downloads finalized 1-minute NIFTY candles;
- forces NIFTY spot volume to `null`;
- validates one live tick through the canonical ingestion layer while NSE is open;
- skips the tick check outside the regular NSE session;
- generates no signal and calls no order endpoint.

Expected final line:

```text
RESULT: ANGEL ONE PROVIDER VERIFICATION PASSED
```

If verification fails, copy only the sanitized error line beginning with
`RESULT:`. Do not copy provider logs or any credentials.

## Step 11 completion gate

Step 11 is complete only when:

- `check:step11` passes;
- all backend tests pass;
- the full Windows test command passes;
- historical NIFTY 1-minute candles are returned with `V=null`;
- a canonical live tick passes during market hours, or is explicitly skipped
  because the market is closed;
- `LIVE_SIGNAL_KILL_SWITCH=true` remains unchanged.

The next step will connect this verified adapter to the backend lifespan,
continuous recovery/finalization service, market-state publication, and live
chart without allowing developing candles into official inference.

# Step 10 — Command Prompt Procedure

These commands install the complete Step 10 package in a new folder while preserving Step 9.

## 1. Stop local servers

In each Command Prompt window running the frontend or backend, press `Ctrl+C`.

## 2. Back up Step 9 and extract Step 10

```text
cd /d C:\Users\sy771\Downloads
ren stock-intelligence stock-intelligence-step9-backup
mkdir stock-intelligence
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\nifty-intelligence-terminal-step-10.zip' -DestinationPath '%USERPROFILE%\Downloads\stock-intelligence' -Force"
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

If that backup name exists, use a unique name such as `stock-intelligence-step9-backup-2`.

## 3. Install dependencies

```text
npm.cmd install
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e "services\backend[server,test,research]"
```

## 4. Create the local environment file

```text
copy .env.example .env
notepad .env
```

Keep these development values:

```text
APP_ENV=development
API_AUTH_MODE=disabled
MARKET_DATA_MODE=replay
MARKET_DATA_PROVIDER=
LIVE_SIGNAL_KILL_SWITCH=true
```

Do not enter a fake provider, model, or calibration value.

## 5. Run complete verification

```text
npm.cmd run lint:windows
npm.cmd run test:windows
npm.cmd run benchmark:backend:windows
```

Expected results include `Step 10 check passed`, `Ran 106 tests`, and `OK`. Exact benchmark times
depend on the computer and measure local API overhead only.

## 6. Start the backend

Open a new Command Prompt:

```text
cd /d C:\Users\sy771\Downloads\stock-intelligence
npm.cmd run backend:dev:windows
```

## 7. Start the frontend

Open a second Command Prompt:

```text
cd /d C:\Users\sy771\Downloads\stock-intelligence
npm.cmd run dev:windows
```

The terminal should show Step 10 of 10, the readiness panel, `LIVE ANALYSIS UNAVAILABLE`, and
blocked release reasons. That is correct without a licensed provider and approved artifacts.

Do not switch to production or clear the kill switch yet. Production requires a provider benchmark,
HTTPS backend proxy/BFF, server-side authentication, validated artifacts, a drift baseline, backup
verification, and explicit approval.

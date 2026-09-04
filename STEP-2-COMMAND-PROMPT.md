# Step 2 — Windows Command Prompt update

This guide covers only Stage 3, Step 2: the canonical market-data layer and
provider abstraction.

The Step 2 archive contains the complete current project. It overwrites tracked
Step 1 source files with their complete Step 2 versions and adds the new files.
It does not contain `.env`, so your local secrets/configuration remain untouched.

## 1. Open Command Prompt

Press `Windows + R`, enter `cmd`, and press Enter.

Move to the existing project:

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
```

Confirm the location:

```bat
cd
dir
```

## 2. Apply the complete Step 2 archive

Download `nifty-intelligence-terminal-step-2.zip` into Downloads, then run:

```bat
set "STEP2_ARCHIVE=%USERPROFILE%\Downloads\nifty-intelligence-terminal-step-2.zip"
set "STEP2_TEMP=%TEMP%\nifty-step2-%RANDOM%"
mkdir "%STEP2_TEMP%"
tar -xf "%STEP2_ARCHIVE%" -C "%STEP2_TEMP%"
robocopy "%STEP2_TEMP%\nifty-intelligence-terminal" "." /E /R:1 /W:1
rmdir /S /Q "%STEP2_TEMP%"
```

`robocopy` exit codes from 0 through 7 are successful copy outcomes. The next
commands can be run normally even if Command Prompt displays `1` as its summary.

## 3. Keep the project in replay mode

Open the local environment file:

```bat
notepad ".env"
```

Ensure it contains these values:

```text
MARKET_DATA_MODE=replay
MARKET_DATA_PROVIDER=
```

Do not add a provider key. Step 2 contains no licensed live adapter.

## 4. Install/update dependencies

```bat
npm.cmd install
```

Step 2 adds no paid dependency and no provider SDK.

## 5. Run Step 2 verification

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run lint:windows
npm.cmd run test:windows
```

Expected results include:

```text
Foundation check passed (7 required paths).
Step 2 check passed (12 required paths).
OK
pass 1
fail 0
```

You can also run only the Python tests:

```bat
set PYTHONPATH=services\backend\src
python -m unittest discover -s services\backend\tests -p "test_*.py"
```

## 6. Start the website

```bat
npm.cmd run dev:windows
```

Open the local URL shown in Command Prompt. Stop the server with `Ctrl+C`.

The Step 2 page must show:

- `Step 2 of 10`
- `Provider adapter — Interface ready`
- `Canonical event — Schema v1`
- `Validation — Fail-safe`
- `Live provider — Not configured`
- `LIVE ANALYSIS UNAVAILABLE`

It must not show live prices, candles, probabilities, signals, accuracy, or performance.

## 7. Complete files added or replaced in Step 2

Open any file with Notepad using these commands:

```bat
notepad "src\config\project.ts"
notepad "app\page.tsx"
notepad "app\globals.css"
notepad ".env.example"
notepad "package.json"
notepad "README.md"
notepad "contracts\README.md"
notepad "contracts\market-event.v1.schema.json"
notepad "contracts\provider-health.v1.schema.json"
notepad "scripts\check-step2.mjs"
notepad "tests\rendered-html.test.mjs"
notepad "services\backend\src\nifty_terminal\settings.py"
notepad "services\backend\src\nifty_terminal\domain\enums.py"
notepad "services\backend\src\nifty_terminal\domain\instruments.py"
notepad "services\backend\src\nifty_terminal\domain\market_event.py"
notepad "services\backend\src\nifty_terminal\providers\base.py"
notepad "services\backend\src\nifty_terminal\providers\replay.py"
notepad "services\backend\src\nifty_terminal\ingestion\normalizer.py"
notepad "services\backend\src\nifty_terminal\ingestion\validator.py"
notepad "services\backend\src\nifty_terminal\ingestion\deduplicator.py"
notepad "services\backend\src\nifty_terminal\ingestion\ledger.py"
notepad "services\backend\src\nifty_terminal\ingestion\pipeline.py"
notepad "services\backend\src\nifty_terminal\ingestion\sequence.py"
notepad "services\backend\tests\test_market_data_domain.py"
notepad "services\backend\tests\test_validation.py"
notepad "services\backend\tests\test_ingestion_pipeline.py"
```

All source in the archive is complete. Do not combine it with partial snippets.

## Step 2 completion condition

Step 2 is complete only when the foundation check, Step 2 structural check,
all backend tests, frontend lint, production build, and rendered-page test pass.
Do not create candles or market-state snapshots until Step 3 is explicitly requested.

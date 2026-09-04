# Stage 3 — Step 6 (Windows Command Prompt)

Step 6 adds volatility-adjusted first-touch labels, purged chronological model
validation, uncalibrated baseline training, and immutable historical
simulated-live replay. It does not create an official model, calibrated
probability, BUY/SELL/WAIT signal, or trading-performance claim.

## 1. Open the project

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd install
```

## 2. Install the research dependencies

If `.venv` already exists:

```bat
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test,research]"
```

If it does not exist:

```bat
python -m venv ".venv"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test,research]"
```

## 3. Verify Steps 1–6

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run check:step4
npm.cmd run check:step5
npm.cmd run check:step6
npm.cmd run lint:windows
npm.cmd run test:windows
```

## 4. Run locally

Backend Command Prompt:

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd run backend:dev:windows
```

Frontend Command Prompt:

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd run dev:windows
```

The chart remains disconnected until licensed live data is configured.

## 5. Optional research run with authorized historical data

Do this only after Step 5 imported a real provider dataset with a `PASS` quality
verdict. Copy the exact `dataset_id` printed by the import command.

```bat
".venv\Scripts\python.exe" -m nifty_terminal.cli.train_replay --database "data\research.sqlite3" --dataset-id "PASTE-DATASET-ID-HERE" --calendar-json "config\nse-calendar.local.json" --output-dir "artifacts\research"
```

The command refuses `DEGRADED` or `REJECTED` datasets and refuses to overwrite
an existing immutable report. Its probabilities are raw research outputs and
are not displayed on the terminal.

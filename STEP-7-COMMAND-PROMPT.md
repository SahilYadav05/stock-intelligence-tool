# Stage 3 — Step 7 (Windows Command Prompt)

Step 7 adds chronological probability calibration, empirical release gates,
and a deterministic WAIT-first signal/risk/lifecycle engine. It does not add a
licensed live provider, approve a model automatically, or enable trading.

## 1. Open and install the project

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd install
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test,research]"
```

If `.venv` does not exist, create it first:

```bat
python -m venv ".venv"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test,research]"
```

## 2. Verify Steps 1–7

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run check:step4
npm.cmd run check:step5
npm.cmd run check:step6
npm.cmd run check:step7
npm.cmd run lint:windows
npm.cmd run test:windows
```

## 3. Run locally

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

The UI remains disconnected and emits no signal until a licensed provider and
an approved model/calibration artifact exist.

## 4. Optional calibration and policy replay

First complete a real Step 6 run. Copy its exact `run_id`, then execute:

```bat
".venv\Scripts\python.exe" -m nifty_terminal.cli.calibrate_policy --database "data\research.sqlite3" --run-id "PASTE-RUN-ID-HERE" --output-dir "artifacts\calibration"
```

The command reads only immutable OOS predictions for the selected Step 6
candidate. It refuses to overwrite an existing report. Failure of any release
gate is a valid result and produces WAIT decisions; it must not be bypassed.

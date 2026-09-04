# Stage 3 — Step 5 (Windows Command Prompt)

Step 5 adds the historical-data import pipeline, append-only SQLite research
store, strict data-quality report, and deterministic multi-timeframe feature
engine. It does not train an ML model or generate a signal.

## 1. Open the project and install the updated packages

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd install
```

## 2. Refresh the isolated Python environment

If `.venv` already exists:

```bat
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test]"
```

If it does not exist:

```bat
python -m venv ".venv"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test]"
```

## 3. Verify all completed layers

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run check:step4
npm.cmd run check:step5
npm.cmd run lint:windows
npm.cmd run test:windows
```

## 4. Run the terminal

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

The chart remains disconnected because Step 5 does not configure licensed data.

## 5. Optional authorized historical import

First read `HISTORICAL-DATA-FORMAT.md`. Do not run this with invented or
unauthorized data.

Prepare a calendar file using official NSE dates:

```bat
copy "config\nse-calendar.example.json" "config\nse-calendar.local.json"
notepad "config\nse-calendar.local.json"
```

Then import your provider-exported CSV, replacing every placeholder:

```bat
".venv\Scripts\python.exe" -m nifty_terminal.cli.import_history --csv "C:\path\to\provider-export.csv" --provider "provider-name" --database "data\research.sqlite3" --starts-at "YYYY-MM-DDTHH:MM:SS+05:30" --ends-at "YYYY-MM-DDTHH:MM:SS+05:30" --calendar-json "config\nse-calendar.local.json"
```

The command prints:

- immutable dataset ID
- source quality status
- missing-minute count
- stored candle revision count
- materialized 5m, 15m, and 1h feature-row counts

`PASS`, `DEGRADED`, and `REJECTED` are distinct outcomes. Missing data is never
forward-filled and a rejected dataset cannot create feature rows.

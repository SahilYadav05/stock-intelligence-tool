# Stage 3 — Step 9 (Windows Command Prompt)

Step 9 adds immutable prediction registration, later outcome assessments,
conservative paper-trade lifecycle events, sample-gated analytics, monitoring,
and the Research, Journal, and Monitoring dashboard sections.

It does not place orders, recommend options, invent market history, or display
performance percentages without sufficient assessed samples.

## 1. Install or update

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd install
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test,research]"
```

If `.venv` does not exist:

```bat
python -m venv ".venv"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test,research]"
```

## 2. Verify Steps 1–9

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run check:step4
npm.cmd run check:step5
npm.cmd run check:step6
npm.cmd run check:step7
npm.cmd run check:step8
npm.cmd run check:step9
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
set NEXT_PUBLIC_MARKET_API_URL=http://127.0.0.1:8000
npm.cmd run dev:windows
```

The tracking panels will show zero observations and unavailable metrics until
real canonical predictions and later outcomes exist. That empty state is the
correct result; do not seed the production interface with invented trades.

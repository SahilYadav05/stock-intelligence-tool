# Stage 3 — Step 8 (Windows Command Prompt)

Step 8 adds the complete professional chart-first dashboard, synchronized
analysis delivery contract, chart overlays, multi-timeframe context panels,
data-quality audit strip, and responsive empty/stale states.

It does not add a licensed provider, approve a model, execute trades, or invent
demonstration market data.

## 1. Open and install the project

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

## 2. Verify Steps 1–8

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run check:step4
npm.cmd run check:step5
npm.cmd run check:step6
npm.cmd run check:step7
npm.cmd run check:step8
npm.cmd run lint:windows
npm.cmd run test:windows
```

## 3. Run the dashboard locally

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

The dashboard will truthfully remain disconnected until the provider adapter
is configured in a later approved integration step. Do not insert fake candles
or manually constructed probabilities to make the screen look active.

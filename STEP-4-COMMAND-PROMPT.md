# Stage 3 — Step 4 (Windows Command Prompt)

Step 4 adds the local backend API, WebSocket market-state stream, and the
interactive Lightweight Charts terminal. It does not connect a licensed market
provider or generate features, predictions, probabilities, or signals.

## 1. Open Command Prompt in the project

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
```

## 2. Install frontend packages

```bat
npm.cmd install
```

## 3. Create the isolated Python environment

```bat
python -m venv ".venv"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e "services\backend[server,test]"
```

## 4. Configure the local browser-to-backend address

```bat
notepad ".env.local"
```

Replace the complete file contents with:

```text
NEXT_PUBLIC_APP_NAME=NIFTY Intelligence Terminal
NEXT_PUBLIC_APP_ENV=development
NEXT_PUBLIC_MARKET_API_URL=http://127.0.0.1:8000
```

Do not place provider credentials in any `NEXT_PUBLIC_` variable.

## 5. Verify all completed steps

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run check:step4
npm.cmd run lint:windows
npm.cmd run test:windows
```

The backend suite should report all tests passing. The complete test command
also creates a production frontend build and validates the rendered Step 4 page.

## 6. Start the backend

Open one Command Prompt window and run:

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd run backend:dev:windows
```

Leave that window open. The API runs only on your computer at
`http://127.0.0.1:8000`.

## 7. Start the website

Open a second Command Prompt window and run:

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
npm.cmd run dev:windows
```

Open the address printed by Vite.

## Expected result

The interactive chart is continuously visible with zoom, pan, and crosshair.
Because no licensed provider is connected, it must show:

- `DISCONNECTED`
- `SYNCING ANALYSIS`
- `NO SIGNAL`
- `Awaiting canonical market data`
- `No prices are being fabricated`

This empty state is correct. Step 4 establishes transport and visualization;
it does not pretend that test data is live.

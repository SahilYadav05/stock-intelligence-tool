# Stage 3 — Step 3 (Windows Command Prompt)

Step 3 adds the NSE session calendar, developing-candle isolation, authoritative
minute finalization, 5m/15m/1h aggregation, append-only corrections, and
versioned market-state snapshots. It does not add a live provider, API,
WebSocket, chart, feature engine, model, probability, or signal.

## 1. Open the project

Run these commands in **Command Prompt**, not PowerShell:

```bat
cd /d "%USERPROFILE%\Downloads\stock-intelligence"
```

## 2. Install or refresh dependencies

```bat
npm.cmd install
```

Using `npm.cmd` bypasses the PowerShell `npm.ps1` execution-policy problem and
also works correctly in Command Prompt.

## 3. Keep provider configuration honest

```bat
notepad ".env"
```

Do not add invented market-data credentials. Keep live-provider fields empty
until a provider has been licensed and benchmarked. Replay fixtures remain test-only.

## 4. Verify every completed layer

```bat
npm.cmd run check:foundation
npm.cmd run check:step2
npm.cmd run check:step3
npm.cmd run lint:windows
npm.cmd run test:windows
```

The backend suite must report 30 passing tests. The complete command also builds
the web application and checks the rendered Step 3 status page.

## 5. Run locally

```bat
npm.cmd run dev:windows
```

Open the local URL printed by Vite. Stop the server with `Ctrl+C`.

## Expected result

The page must show:

- `Step 3 of 10`
- `Developing candle — Visual only`
- finalized `5m · 15m · 1h` context
- `LIVE ANALYSIS UNAVAILABLE`

That last status is intentional: no licensed live provider is configured in Step 3.

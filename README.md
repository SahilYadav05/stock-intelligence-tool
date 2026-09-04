# NIFTY Intelligence Terminal

A private, evidence-first trading-intelligence and quantitative-research terminal.

## Current state

Stage 4, Step 22 adds append-only NIFTY derivatives context collection to the
existing live/replay pipeline, cross-market research, shadow inference,
calibration, tracking, drift controls, and baseline-controlled validation.
The terminal can now show a finalized-candle conditional setup with a trigger,
structure-aware stop/invalidation, and targets at 1.25R, 2R and 3R.

The completed Step 19 historical experiment did **not** establish a stable edge,
so no probabilistic model is approved and the official state remains WAIT.
Price-action output is explicitly research-only and cannot place orders. The
experiment kept the Step 18F gates unchanged and found no policy that passed
selection; genuinely future confirmation is still mandatory before model release.

The corrected Step 20 model was also rejected. On the expanded 2024-2026 data it
generated 918 diagnostic trades, but only a 46.95% win rate, 1.06 profit factor,
and 31.55R drawdown. Step 21 then tested 6,220 causal price-action events with a
five-trade daily cap. That result was worse: 39.87% wins, 0.72 profit factor and
51.25R drawdown. These negative experiments remain immutable evidence; their
thresholds will not be retuned against the diagnostic folds.

The verified research base now spans 2024-01-01 through 2026-08-25 with 244,305
regular-session NIFTY minutes, 100% expected-minute coverage after explicit
quarantines, and 40,497 eligible labels. The next legitimate alpha workstream is
collection of point-in-time derivatives volume/open interest, option sentiment,
and breadth rather than further price-only tuning. Futures/OI and option context
are now captured by a read-only hashed ledger, but remain unavailable to model
inference until enough untouched sessions have accumulated.

## Repository layout

```text
app/                  Web application routes and global styling
src/components/       Interactive chart terminal and connection states
src/lib/              Browser-side versioned transport types and validation
services/backend/     Python domain, history, features, delivery API, and tests
contracts/            Versioned market, history, feature, and transport contracts
scripts/              Repository verification helpers
tests/                Web build/render verification
docs/                 Short implementation records and Windows instructions
```

## Prerequisites

- Node.js 22.13 or newer
- npm 10 or newer
- Python 3.12 or newer
- Git

## Local frontend

```text
npm install
npm run dev:windows
```

Open the local address printed by the development server.

## Local backend

Create and install the isolated Python environment first:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e "services\backend[server,test,research]"
npm.cmd run backend:dev:windows
```

The API binds to `127.0.0.1:8000`. It reports `DISCONNECTED` until a licensed
provider is configured in a later approved step.

## Backend tests (Command Prompt)

```text
npm.cmd run test:backend:windows
```

## Verification

```text
npm run check:foundation
npm run check:step2
npm run check:step3
npm run check:step4
npm run check:step5
npm run check:step6
npm run check:step7
npm run check:step8
npm run check:step9
npm run check:step10
npm run check:step19
npm run check:step20
npm run check:step21
npm run check:step22
npm run lint:windows
npm run test:windows
```

## Security boundary

- Provider credentials and bearer secrets remain server-side.
- `.env` files are ignored and must never be committed.
- The browser will never connect directly to a market-data provider.
- No order-placement capability is part of the MVP.
- Conditional price-action plans are not calibrated probabilities or official signals.
- Provider selection remains adapter-based and contract-gated.
- Production startup fails closed without HTTPS origins and authentication.
- Live release requires artifact hashes, calibration evidence, drift evidence,
  exact snapshot parity, and a cleared operator kill switch.

See `STEP-10-COMMAND-PROMPT.md` for the exact Windows procedure and
`SECURITY-DEPLOYMENT-RUNBOOK.md` for production boundaries and release gates.

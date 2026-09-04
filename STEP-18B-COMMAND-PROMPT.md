# Step 18B — Complete Trade-Aligned Model Improvement

Step 18B replaces no live or shadow artifact. It is a fail-closed research revision
created because Step 18 was not strong enough.

Research version `trade_aligned_model_research.v1.1` also disqualifies any model
fold whose optimizer does not converge. Optimizer warnings can never be silently
accepted as valid candidate probabilities.

## 1. Stop local processes

Press `Ctrl+C` in Command Prompt windows running the backend or frontend.

## 2. Open the project

```bat
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

## 3. Create a backup

```bat
tar -a -c -f pre-step18b-backup.zip package.json contracts scripts services\backend\src\nifty_terminal services\backend\tests
```

Do not include `.env`, databases, provider credentials or research artifacts in files
you upload or share.

## 4. Install the Step 18B package

Place `step18b-complete-model-improvement.zip` in Downloads, then run:

```bat
tar -xf "%USERPROFILE%\Downloads\step18b-complete-model-improvement.zip" -C .
```

## 5. Run the structural check

```bat
npm.cmd run check:step18b
```

Expected:

```text
Step 18B check passed (6 required paths).
```

## 6. Run all regression tests

```bat
npm.cmd run test:windows
```

The existing Starlette/httpx warning is not a failure. Stop and send the complete
output if a test reports `FAILED` or `ERROR`.

## 7. Run the complete improved research

Keep this fail-safe setting in `.env`:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

Run:

```bat
npm.cmd run research:trade-aligned:windows
```

Allow approximately 20–60 minutes on a local CPU. The command performs one complete
research run containing:

- separate long and short target-before-stop labels;
- stationary multi-timeframe price features;
- normalized MACD, ADX, slopes and volatility;
- opening gap/range, previous-session levels and rolling support/resistance;
- causal numeric candlestick-pattern features;
- regularized logistic, elastic-net and gradient-boosting comparisons;
- historical-prior and simple technical baselines;
- binary calibration comparison;
- purged chronological validation;
- five-fold stability gates;
- session-block bootstrap confidence intervals;
- cost-aware BUY/SELL/WAIT simulated-live replay;
- independent BUY and SELL support gates.

Expected final lines:

```text
RESULT: STEP 18B TRADE-ALIGNED RESEARCH COMPLETED
No model or signal was released. A failed gate is a valid result.
```

The immutable report is written to:

```text
artifacts\research\trade-aligned-v3\<experiment-id>.json
```

Send the complete terminal output and upload the generated JSON report. Do not adjust
thresholds merely to force a pass.

## Important limitations

This step does not fabricate data that is not available. Historical news, India VIX,
NIFTY futures volume/open interest and constituent breadth remain excluded until each
has a legitimate timestamped canonical dataset. NIFTY spot volume and VWAP remain
unavailable.

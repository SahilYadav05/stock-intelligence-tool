# Step 18 — Enhanced Features + Hierarchical Model-V2 Research

This step is research-only. It does not replace the Step 17 shadow model, policy,
manifest or ledger. It cannot enable precise live probabilities, official signals or
automatic trading.

## 1. Stop local processes before installing

In every Command Prompt running the frontend or backend, press `Ctrl+C`.

## 2. Open the project directory

```bat
cd /d C:\Users\sy771\Downloads\stock-intelligence
```

## 3. Create a backup

```bat
tar -a -c -f pre-step18-backup.zip package.json contracts scripts services\backend\src\nifty_terminal services\backend\tests
```

Do not include `.env`, provider credentials, databases or research artifacts in files
you share.

## 4. Install the Step 18 package

Place `step18-hierarchical-model-v2-research.zip` in Downloads, then run:

```bat
tar -xf "%USERPROFILE%\Downloads\step18-hierarchical-model-v2-research.zip" -C .
```

## 5. Run the structural check

```bat
npm.cmd run check:step18
```

Expected final line:

```text
Step 18 check passed (6 required paths).
```

## 6. Run the complete regression suite

```bat
npm.cmd run test:windows
```

The existing Starlette/httpx deprecation warning is not a test failure. Stop and send
the complete output if any test reports `FAILED` or `ERROR`.

## 7. Run model-v2 research

Keep this in `.env`:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

Then run:

```bat
npm.cmd run research:model-v2:windows
```

This may take 10–30 minutes on a local CPU. It compares the old direct classifier,
enhanced direct classifiers and hierarchical opportunity/direction classifiers. It
uses purged chronological folds and rejects BUY-only or SELL-only collapse.

Expected final lines:

```text
RESULT: STEP 18 MODEL-V2 RESEARCH COMPLETED
No model or signal was released; inspect the research gates before Step 19.
```

The complete immutable report is written under:

```text
artifacts\research\model-v2\<experiment-id>.json
```

Send the terminal JSON output before proceeding to Step 19. A failed research gate is
a valid, safe result and must not be bypassed by lowering thresholds.

## 8. Resume the Step 17 forward shadow collector

After research finishes, start the backend normally during a future NSE session:

```bat
npm.cmd run backend:dev:windows
```

Step 18 does not alter `SHADOW_RUNTIME_MANIFEST_PATH` or `SHADOW_LEDGER_PATH`.

# Step 17 — Signal-policy research and automatic live shadow observation

Step 16 proved that the model's probability scores slightly beat the historical
prior, but its original 60% signal policy produced zero trades. Step 17 does
not lower one threshold until calls appear. It performs a locked chronological
policy experiment and then connects the resulting artifacts to the same
canonical finalized-candle snapshots used by the chart.

This step adds:

- 432 predefined policies across raw directional scores and calibrated
  probabilities;
- policy selection on chronological fold 3 only;
- locked historical evaluation on later fold 4 only;
- one-minute entry, stop, target and expiry simulation with slippage;
- minimum trade and BUY/SELL support gates;
- hash-verified model, policy and runtime manifests;
- an append-only SQLite shadow prediction and assessment ledger;
- automatic inference after a new finalized live 5-minute snapshot;
- automatic 60-minute UP/DOWN/NEITHER outcome assessment;
- a safe shadow status endpoint that exposes counts but no probabilities.

If no policy clears the predefined gates, the runtime uses `WAIT_ONLY`. It still
records probabilities and later outcomes so genuinely unseen evidence can
accumulate. It never forces a BUY or SELL.

## Apply

Stop the backend and frontend. Download the Step 17 ZIP into Downloads, then
run in Command Prompt:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step17-backup.zip package.json .env.example contracts scripts services\backend\src\nifty_terminal services\backend\tests
tar -xf "%USERPROFILE%\Downloads\step17-shadow-policy-runtime.zip" -C .
npm.cmd run check:step17
set PYTHONPATH=services/backend/src;services/backend/tests&& .venv\Scripts\python.exe -m unittest test_probability_research_v2 test_locked_shadow_research test_shadow_runtime
npm.cmd run test:windows
```

Do not continue if a command fails. Do not replace your `.env` with
`.env.example`; the real Angel One credentials already belong only in `.env`.

## Run policy research

Keep this setting:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

Run:

```cmd
npm.cmd run research:shadow-policy:windows
```

The command may take several minutes. It automatically uses the newest
immutable Step 16 report and validates its model SHA-256 before research.

Successful completion ends with:

```text
RESULT: STEP 17 POLICY RESEARCH COMPLETED
The runtime manifest is shadow-only; official signals remain disabled.
```

## Enable prediction collection

Copy the exact `shadow_runtime_manifest.path` printed by the research command.
Open `.env`:

```cmd
notepad .env
```

Add or replace these three lines:

```text
SHADOW_MODE_ENABLED=true
SHADOW_RUNTIME_MANIFEST_PATH=PASTE_THE_PRINTED_MANIFEST_PATH_HERE
SHADOW_LEDGER_PATH=data\shadow-ledger.sqlite3
```

Example only—do not copy this filename unless it is exactly what your command
printed:

```text
SHADOW_RUNTIME_MANIFEST_PATH=artifacts\shadow-runtime\abc123.json
```

Do not change:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

Verify the complete artifact chain and initialize the append-only ledger:

```cmd
npm.cmd run verify:shadow-runtime:windows
```

Expected ending:

```text
RESULT: SHADOW RUNTIME VERIFICATION PASSED
Official signals, precise probability display and trading remain disabled.
```

## Start automatic observation

Start the backend in one Command Prompt:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
.venv\Scripts\activate.bat
npm.cmd run backend:dev:windows
```

Start the frontend in another Command Prompt:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
npm.cmd run dev:windows
```

Check safe shadow status:

```cmd
curl http://127.0.0.1:8000/api/v1/shadow/status
```

During an open NSE continuous session, each genuinely new finalized 5-minute
snapshot can add one immutable shadow prediction. After its 60-minute outcome
window becomes available, a separate immutable assessment is added. Developing
candles never enter this process. Provider corrections create new snapshot
identities and never rewrite an old prediction.

The terminal continues to display WAIT/no approved analysis. Raw and calibrated
probabilities remain hidden from the user-facing dashboard until forward
calibration and release gates pass.

## Important operating limitation

Shadow collection runs only while the Python backend is running and connected
to Angel One. The current Cloudflare Worker hello-world does not run this
stateful Python/WebSocket process. Always-on forward collection will require a
later backend deployment or a continuously running local machine.

News is not included in Step 17. It will be added as a separately timestamped
event-risk layer only after the market-only shadow pipeline is producing
auditable forward observations.

## Share back

Share the printed Step 17 summary containing:

```text
dataset_id
experiment_id
candidate_count
passing_selection_candidate_count
selected_policy
selection_metrics
selection_blockers
historical_evaluation
historical_evaluation_blockers
historical_policy_gate_passed
policy_artifact
shadow_runtime_manifest
report_path
```

Do not share `.env`, Angel One credentials, SQLite files, raw licensed candles,
individual hidden probabilities, or the detailed trade ledger.

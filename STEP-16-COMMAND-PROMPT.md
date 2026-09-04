# Step 16 — Locked shadow candidate and simulated-live backtest

Step 15 found that the 1.5 ATR target had the strongest Brier skill relative
to its own target-specific prior. Its absolute Brier score must not be compared
directly with the absolute Brier score of a different label definition. This
step corrects that ranking defect and locks the research specification to:

- symmetric 1.5 ATR UP/DOWN barriers;
- a 60-minute horizon;
- unweighted multinomial logistic regression;
- finalized 5-minute decisions with finalized 15-minute and 1-hour context;
- no developing candles, NIFTY spot volume, VWAP or news.

It then performs a chronology-preserving calibration comparison, constructs a
safe JSON shadow artifact, and runs a signal-policy replay on one-minute bars.
It still cannot approve the model for official live use: the target was chosen
using this historical period, so genuinely later forward confirmation is
mandatory.

## What the backtest does

- Uses only the last chronological fold for this historical diagnostic.
- Enters at the next available one-minute open after a finalized 5-minute
  decision.
- Uses a fixed WAIT-heavy policy: 60% directional probability, 15-point
  probability margin and maximum 45% NEITHER probability.
- Uses a 0.75 ATR stop, 1.0 ATR target and 60-minute expiry.
- Resolves a stop and target touched within the same minute as STOP first.
- Applies 0.5 NIFTY points of slippage on both entry and exit.
- Allows only one active position at a time.
- Reports hypothetical NIFTY index points and R multiples only. NIFTY spot is
  not directly tradable, so this is not rupee P&L and does not claim brokerage,
  taxes, futures basis or execution realism.

This is a simulated-live event replay, not a random train/test split and not a
promise of future profitability.

## Apply

Stop the backend and frontend. In Command Prompt run:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step16-backup.zip package.json contracts scripts services\backend\src\nifty_terminal services\backend\tests
tar -xf "%USERPROFILE%\Downloads\step16-locked-shadow-backtest.zip" -C .
npm.cmd run check:step16
set PYTHONPATH=services/backend/src;services/backend/tests&& .venv\Scripts\python.exe -m unittest test_probability_research_v2 test_locked_shadow_research
npm.cmd run test:windows
```

Do not proceed if a check fails. The `.env` must continue to contain:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

No new credential or environment variable is required.

## Run

```cmd
npm.cmd run research:locked-shadow:windows
```

The command may take several minutes on a local CPU. It prints a compact
summary and writes the detailed immutable trade ledger to the report file.

Successful execution ends with:

```text
RESULT: STEP 16 LOCKED SHADOW RESEARCH COMPLETED
No model was released for official live inference or order execution.
```

## Share back

Share the printed JSON summary containing:

```text
dataset_id
experiment_id
locked_specification
selected_calibration_method
historical_backtest_probability_metrics
signal_backtest
shadow_artifact
report_path
```

Do not share `.env`, the SQLite database, provider credentials, licensed raw
candles, or the detailed trade ledger.

## Release interpretation

- A profitable historical signal replay does not override probability gates.
- `approved_for_live_inference` remains false by design.
- Precise percentages and official BUY/SELL signals remain suppressed.
- The JSON artifact is safe to inspect and hash-check, but it is marked
  `shadow_only` and cannot be loaded as an approved production model.
- Step 17 will connect this artifact to live finalized snapshots in shadow mode
  and accumulate immutable forward predictions and later outcomes.
- Any change to the target, features, model, calibration method or signal
  thresholds restarts forward confirmation.
- News will be introduced later as a separately timestamped context/risk layer;
  it will not rewrite the model's numeric probabilities.

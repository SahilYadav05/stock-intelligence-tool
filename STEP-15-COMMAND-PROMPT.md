# Step 15 — Probability and target research v2

Step 14 proved that the balanced-class model improved balanced accuracy but
produced worse probability scores than a historical-prior baseline. This step
does not hide that result or deploy the failed model.

Step 15 performs one larger screening experiment:

- symmetric 1.0, 1.25 and 1.5 ATR barriers over the locked 60-minute horizon;
- unweighted and balanced multinomial logistic regression;
- unweighted and balanced histogram gradient boosting;
- folds 0–2 for candidate selection only;
- fold 3 for temperature fitting only;
- fold 4 for later final screening only;
- Brier skill, log loss, calibration error, class support and balanced accuracy;
- immutable JSON output with no deployable artifact.

Because the three target definitions are compared using the final screening
fold, even the screening leader must later pass a locked confirmation run. Step
15 cannot approve live inference.

## Apply

Stop the backend and run:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step15-backup.zip package.json contracts scripts services\backend\src\nifty_terminal services\backend\tests
tar -xf "%USERPROFILE%\Downloads\step15-probability-research-v2.zip" -C .
npm.cmd run check:step15
npm.cmd run test:windows
```

Do not continue unless both checks pass. Keep:

```text
LIVE_SIGNAL_KILL_SWITCH=true
```

## Run

```cmd
npm.cmd run research:probability-v2:windows
```

Expected local CPU runtime is approximately 10–30 minutes. Progress is printed
once for each ATR target. Do not stop the command merely because individual
model fits are quiet.

Completion ends with:

```text
RESULT: STEP 15 PROBABILITY RESEARCH COMPLETED
No model, precise live probability, official signal or order was released.
```

Share these safe sections:

```text
dataset_id
experiment_id
targets
screening_leader
report_path
```

Do not share `.env`, the database, raw licensed candles or provider tokens.

## Interpretation

- `screening_gate_passed: false` is an honest result, not a program failure.
- Never lower gates to make a candidate pass.
- A passing screening leader is still not live-approved because target choice
  used the final screening results.
- News remains excluded. It will later enter a separately timestamped event-risk
  layer and cannot rewrite numerical probabilities.
- NIFTY spot volume and VWAP remain unavailable rather than fabricated.

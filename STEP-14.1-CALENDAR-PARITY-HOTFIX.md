# Step 14.1 — Sourced-calendar parity hotfix

The first Step 14 attempt failed before training because the research CLI used
the built-in weekday calendar instead of the sourced calendar file used by
Step 13.3. The historical dataset contains explicitly authorized Saturday and
special-session candles, so every historical and live computation must load the
same calendar rules.

No training run, probability, signal or order was created by the failed attempt.

## Apply the hotfix

Stop the backend and run:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step14.1-backup.zip services\backend\src\nifty_terminal\cli\run_real_data_research.py services\backend\tests\test_real_data_research.py scripts\check-step14.mjs
tar -xf "%USERPROFILE%\Downloads\step14.1-calendar-parity-hotfix.zip" -C .
npm.cmd run check:step14
npm.cmd run test:windows
```

Do not continue unless both checks pass.

Then rerun the unchanged command:

```cmd
npm.cmd run research:real-data:windows
```

The startup output must now include:

```text
sourced exchange calendar: config\nse-calendar-through-2026-08-25.json
```

Training can take several minutes. The safety switch must remain true.

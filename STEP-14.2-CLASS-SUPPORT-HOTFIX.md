# Step 14.2 — Training support versus evidence-gate hotfix

The Step 14 calendar parity fix worked. The next run reached chronological fold
construction and then exposed a separate configuration error: the preliminary
model-fit support floor was incorrectly set to 500 observations per class in
every early fold.

That value belongs to evidence evaluation, not to whether research is allowed
to run. A rare NEITHER class should produce an honest final blocker rather than
crash the experiment before metrics exist.

This hotfix uses the established computational floor of 25 examples per class
inside every training fold. The independent final research gate remains strict:
it still requires at least 1,000 eligible examples of every target class, 10,000
out-of-sample predictions, fold stability, positive Brier skill, better log
loss and a passing calibration gate.

No threshold used for BUY/SELL activation is changed. Live inference, official
signals and automatic trading remain disabled.

## Apply

Stop the backend and run:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
tar -a -c -f pre-step14.2-backup.zip services\backend\src\nifty_terminal\cli\run_real_data_research.py services\backend\tests\test_real_data_research.py scripts\check-step14.mjs
tar -xf "%USERPROFILE%\Downloads\step14.2-class-support-hotfix.zip" -C .
npm.cmd run check:step14
npm.cmd run test:windows
```

Do not continue unless both commands pass. Then run:

```cmd
npm.cmd run research:real-data:windows
```

The two failed attempts stopped before a training run was persisted, so history
does not need to be downloaded again and no cleanup is required.

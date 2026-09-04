# Step 26 — live-plan-aligned price-action meta-label research

Step 26 corrects the largest mismatch left by the earlier studies.  Direction,
trigger, entry zone, structure stop, invalidation and targets at 1.25R, 2R and
3R now come from the same production price-action engine used by the terminal.
The model cannot reverse that direction; it can only accept the setup or return
WAIT.

The shared minute replay applies 0.5 NIFTY-point slippage on entry and every
exit, does not chase a gap outside the displayed entry zone, resolves a
same-minute stop before targets, and applies a protective stop change only from
the following minute.  Three exit policies are declared in advance: full exit
at T1, 50/30/20 scale-out with a static stop, and 50/30/20 scale-out with
protection after T1 and T2.

Exit selection, compact meta-model selection, calibration, activation-policy
selection and historical diagnosis occupy separate purged chronological folds.
The selected compact model uses only the 25-field price-action and structure
family.  Positions cannot overlap and no more than five trades can be opened in
one session.  Win rate is reported with a Wilson interval and cannot pass
without positive bootstrapped expectancy, profit factor, drawdown and session
support gates.

## Release boundary

This history has already influenced development, the replay remains a NIFTY
spot proxy rather than an executable futures contract, and the forward
derivatives ledger is not ready.  The experiment can reject an approach but
cannot create a live artifact or official signal.  A historical pass would
still require frozen confirmation on genuinely future futures data after all
costs.

## Completed result

The completed experiment result is recorded after the pinned-data run.

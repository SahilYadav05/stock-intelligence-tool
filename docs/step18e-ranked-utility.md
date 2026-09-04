# Step 18E — Ranked utility cohort research

Step 18E corrects the policy mismatch discovered by Step 18D. The expected-R
models had small, statistically positive out-of-sample ranking skill, but an
individual 20th-percentile conformal bound for a bounded stop/target outcome is
normally close to the stop. It therefore rejected every observation, even when
the ordering carried information.

This step does not weaken a gate. It changes the uncertainty object:

- Models continue to predict realized LONG and SHORT R under the fixed execution
  simulation.
- Every direction independently compares four feature architectures: NIFTY-only,
  NIFTY plus Bank Nifty, NIFTY plus India VIX, and all available context.
- Model and architecture selection use chronological folds 0 and 1 only.
- Fold 2 supplies the historical score distribution.
- Fold 3 selects a ranked signal policy.
- Fold 4 is a later historical diagnostic with the policy frozen.
- Percentile scores are causal: a decision uses only reference and earlier scores.
- Economic uncertainty is computed for the selected cohort using a session-block
  bootstrap, preserving intraday dependence.

The replay retains next-minute entry, 1.0 ATR target, 0.75 ATR stop, 60-minute
horizon, conservative stop-first resolution, 0.5 NIFTY-point slippage each way,
and no overlapping positions.

## Release posture

This research remains permanently fail-closed. It does not create a deployable
model, enable precise probability display, change the Step 17 shadow runtime,
enable an official signal, or enable automatic trading. Even a clean historical
result still requires genuinely future forward confirmation because the entire
available historical period has already influenced research decisions.

## Interpreting the result

The important fields are:

- `selected_models`: independently chosen model and feature architecture for
  LONG and SHORT.
- `utility_diagnostics`: point-prediction skill, bootstrap skill interval, and
  realized-R deciles on the final historical fold.
- `policy_selection.selected`: the policy chosen on fold 3 only.
- `historical_simulated_live_replay`: frozen-policy results on fold 4.
- `research_gate`: every blocker. Do not delete or relax blockers merely to obtain
  signals.

If no policy passes, the correct result remains WAIT. The next scientifically
useful improvement would require genuinely new information (for example licensed
point-in-time news, breadth, or futures volume/open interest) or new forward data,
not more threshold searching on the same period.

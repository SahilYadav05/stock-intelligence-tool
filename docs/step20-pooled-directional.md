# Step 20 — pooled directional model and trade-cadence gate

Step 19 is not suitable for release. Its separate LONG and SHORT regressors
changed behavior between chronological folds, and its strongest rejected setup
produced only 59 trades across 22 sessions. A good-looking win rate in that small
cohort is not enough evidence.

Step 20 makes one pre-declared methodological change: LONG and SHORT target-first
examples are pooled into a single binary meta-label model. Each decision contributes
both directions to the same chronological fold, with an explicit side interaction;
the model therefore learns reusable setup quality instead of fitting two lower-support
directional functions. The feature candidates are compact, domain-selected variants
of the stationary multi-timeframe, cross-market, and causal price-action inputs.

Seven expanding walk-forward folds are used with a 12-bar purge and embargo:

- folds 0–2 select among three fixed model candidates;
- fold 3 selects identity or Platt calibration and supplies the first score reference;
- fold 4 selects among nine fixed activation/cadence policies;
- folds 5–6 form one larger historical diagnostic period.

The policy gate requires both directions, at least three non-overlapping trades per
represented session, a positive session-bootstrap expectancy bound, an adequate win
rate and profit factor, controlled drawdown and broad session support. The diagnostic
also has to beat WAIT, always-long, always-short, and technical-trend baselines on
the lower 95% daily-R uplift bound. Win rate is never optimized without expectancy.

This historical period has already influenced development. Even a full pass cannot
release an official model; it can only produce a frozen candidate for genuinely
future shadow confirmation.

Version 1.1 adds two non-negotiable invariants found during the first audit run.
Calibration must be monotonic and cannot reverse the model's ranking, and a policy
cannot pass selection when its underlying model failed the model-stability gate.

## Completed expanded-data result

Experiment `86e4b450-fab4-5921-94d8-44026afb5a76` used the verified 2024-01-01
through 2026-08-25 dataset. The diagnostic model showed weak ranking information
(AUC 0.5443 and positive Brier skill), but no model or policy passed selection.
The best rejected policy produced 918 diagnostic trades across 67 sessions, with
a 46.95% win rate, 1.06 profit factor, +0.023R average, and 31.55R maximum
drawdown. Its bootstrap expectancy interval crossed zero and it did not establish
positive uplift over the required baselines. No artifact or official signal was
released.

# Step 25 — compact price-action-first feature audit

This step tests whether redundant indicator families are obscuring the weak
directional signal. It compares five predeclared, nested feature groups using one
strongly regularized logistic direction model:

- the 12 causal price-action features alone;
- price action plus swing levels, opening range, previous-session levels, short
  slope, ADX and directional movement;
- that compact structure set plus Bank Nifty/VIX cross-market relationships;
- that set plus only two higher-timeframe trend transforms;
- the former 45-field structure set, including candle-pattern and MACD fields.

The audit selects architecture on the first three purged folds and checks the chosen
family once on fold four. It does not create another policy backtest or reuse the
final two historical diagnostic folds for tuning. Any selected family remains
research-only and will be augmented with the new derivatives context only after the
forward-data readiness gate passes.

## Completed result

Experiment `5cb04524-1363-547e-82ec-3a2c91e8679b` found:

- `PRICE_ACTION_12`: selection AUC 0.5065; negative Brier skill.
- `STRUCTURE_LEVELS_COMPACT` (25 fields): selection AUC 0.5182 with all three
  fold AUCs above 0.51, but slightly negative Brier skill.
- `STRUCTURE_PLUS_CROSS_MARKET` (31 fields): selection AUC 0.5179; negative
  Brier skill.
- `STRUCTURE_PLUS_HIGHER_TIMEFRAME` (35 fields): selection AUC 0.5303 and
  positive Brier skill, but confirmation AUC fell to 0.4776 with negative skill.
- `LEGACY_STRUCTURE_45`: selection AUC 0.5157; negative Brier skill.

No family passed both selection and chronological confirmation. Pure price action
was not superior, while the old larger feature set also added noise. Accordingly,
the next forward experiment will be price-action-first, cap historical price/context
features at the 35-field family, omit the legacy MACD and candle-pattern clutter,
and add the independently collected derivatives features. The family must be chosen
again using future data; this failed confirmation is not used to manufacture a new
historical policy result.

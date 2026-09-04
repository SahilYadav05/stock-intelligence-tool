# Step 21 — event-based price-action research

Step 20 showed weak directional ranking but activated on too many ordinary
five-minute rows. Step 21 changes the sampling unit, not the historical answer:
it ranks only four predeclared, causal setup families.

- liquidity-sweep reversal around a confirmed swing;
- the first confirmed break of structure;
- compression followed by directional range expansion;
- EMA20 reclaim aligned with confirmed swing structure.

Every feature is available at the finalized decision candle. A previous-row
condition is allowed only within the same exchange session. Opposing events at
one timestamp become `WAIT`, positions cannot overlap, and at most five trades
may be opened in one session. The same seven-fold purged chronology is retained;
the final two folds are diagnostic only.

## Completed expanded-data result

- Dataset: `58e4955e-61e9-587d-b10c-1a7d37736993`
- Context SHA-256: `9452e3e658bfb9c4b78982150c52b9786ef6acc97cd5367e4521e678521fb08b`
- Experiment: `54e421e9-59bd-5d5a-b341-c8dc7a2251a8`
- Complete paths: 39,555; causal events: 6,220.
- Diagnostic: 301 trades over 67 sessions (4.49/session), 39.87% win rate,
  0.72 profit factor, -0.156R average, and 51.25R maximum drawdown.
- Diagnostic event-model AUC was 0.5235 and Brier skill versus the causal prior
  was negative.

No setup, policy, model, or live signal passed. The event definitions must not
be retuned against these diagnostic folds. The next legitimate information
workstream is append-only collection of derivatives volume/open interest,
options sentiment, and point-in-time breadth for future evaluation.


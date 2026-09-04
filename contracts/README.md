# Shared contracts

This directory is the language-neutral boundary between the browser, backend,
market-data adapters, research pipeline, and inference services.

Step 2 introduced `market-event.v1.schema.json` and
`provider-health.v1.schema.json`. Step 3 adds `candle.v1.schema.json` and
`market-state-snapshot.v1.schema.json`.

The candle contract makes developing/finalized state and immutable revision
lineage explicit. The snapshot contract is the shared point-in-time identity
that the chart and future model must both reference.

Step 4 adds `market-state-view.v1.schema.json` and
`websocket-message.v1.schema.json`. A market-state view atomically carries the
snapshot plus the exact finalized and optional developing candle revisions
needed by the browser. The server rejects mismatches before publication.

Step 5 adds `historical-dataset.v1.schema.json` and
`feature-snapshot.v1.schema.json`. Historical imports retain provider identity,
source checksum, acquisition window, and quality verdict. Feature snapshots
retain the exact market snapshot, candle revision checksum, feature version,
and formula-set hash used to calculate them.

Step 6 adds `first-touch-label.v1.schema.json`, `ml-training-run.v1.schema.json`,
`replay-prediction.v1.schema.json`, and `replay-assessment.v1.schema.json`.
Labels lock the 60-minute symmetric ATR first-touch target. Simulated-live raw
predictions remain immutable and uncalibrated; actual outcomes are separate,
later assessment records.

Step 7 adds `calibration-run.v1.schema.json`,
`calibrated-prediction.v1.schema.json`, `signal-decision.v1.schema.json`, and
`signal-lifecycle-event.v1.schema.json`. Calibration is fit on earlier OOS
predictions and judged on a later untouched block. Signals reference the exact
source prediction, calibration, and snapshot; lifecycle changes are new events.

Step 8 adds `analysis-view.v1.schema.json` and
`analysis-availability.v1.schema.json`. The browser requests analysis for the
exact chart snapshot and candle-revision checksum. Mismatch or unavailability
returns `SYNCING_ANALYSIS` and suppresses the signal and chart overlays.

Step 9 adds `tracked-prediction.v1.schema.json`,
`prediction-assessment.v1.schema.json`, `paper-trade.v1.schema.json`,
`paper-trade-event.v1.schema.json`, `monitoring-view.v1.schema.json`, and
`tracking-overview.v1.schema.json`. Original predictions and paper plans are
immutable. Outcomes, fills, exits, and invalidations are later append-only
events. Metrics remain null until their explicit minimum-sample gate passes.

Step 10 adds `artifact-manifest.v1.schema.json`,
`release-readiness.v1.schema.json`, `drift-evidence.v1.schema.json`, and
`security-audit-event.v1.schema.json`.
Readiness is blocked unless artifact identities, calibration evidence, drift
evidence, market state, snapshot parity, and the operator kill switch all pass.

Step 19 adds `price-action-analysis.v1.schema.json`. It binds causal market
structure, support/resistance and a conditional multi-target plan to the exact
snapshot checksum while permanently declaring the output research-only,
uncalibrated and non-executable.

Rules for future contracts:

1. Every contract has an explicit schema version.
2. Provider-specific names and tokens never become canonical identifiers.
3. Timestamps state their semantics and timezone.
4. Developing and finalized candles are different states.
5. Snapshot and candle revision identifiers are mandatory for analysis output.
6. Breaking changes create a new schema version rather than silently changing history.

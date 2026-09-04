# Market delivery backend

The backend exposes two local delivery paths:

- HTTP bootstrap: `GET /api/v1/market-state/NIFTY50_SPOT?timeframe=5m`
- WebSocket updates: `/ws/v1/market-state?instrument_id=NIFTY50_SPOT&timeframe=5m`

It also exposes health and instrument metadata endpoints under `/api/v1`.

Step 19 adds snapshot-bound decision support at
`GET /api/v1/price-action/NIFTY50_SPOT?snapshot_id=...&timeframe=5m`. It uses
only finalized candles and returns confirmed structure, levels, evidence and a
conditional trigger/stop/T1/T2/T3 plan. It never returns a probability or an
official/executable signal.

The delivery service accepts only a validated `MarketStateView`. That view
contains one immutable snapshot plus every finalized candle revision named as
a model input. If a revision is missing, mismatched, developing, or from the
wrong decision time, publication fails before either the chart or a future
model can receive it.

Step 8 adds `analysis-view.v1`, an in-memory read model, and an analysis endpoint
that returns a view only for the exact chart snapshot and candle revision. The
browser suppresses analysis when either identity differs.

Step 9 adds `GET /api/v1/tracking/NIFTY50_SPOT?timeframe=5m`, immutable
prediction/outcome records, append-only paper lifecycle records, evidence-gated
analytics, and an operational/model monitoring view. SQLite tracking tables
reject updates and deletes. Paper results are normalized index points and never
represent an executable NIFTY product or cash performance.

Step 10 adds `/api/v1/live`, fail-closed `/api/v1/ready`, sanitized security
status, exact-origin checks, optional constant-time bearer authentication,
bounded requests/connections, explicit PSI/JSD drift evidence, artifact-hash
release manifests, a circuit breaker, and a hash-chained audit ledger.

There is intentionally no provider connection in Step 10. An empty service
returns HTTP 503 for market state and sends a `DISCONNECTED` WebSocket status.
The browser never connects directly to a provider.

For local development, bind the API to `127.0.0.1`; do not expose it publicly.
Production configuration rejects disabled authentication and non-HTTPS origins.

Install the research environment on Windows with:

```text
.venv\Scripts\python.exe -m pip install -e "services\backend[server,test,research]"
```

No probability is approved for display until ECE, Brier skill, class/bin
support, and chronological stability gates pass. No licensed provider or
approved artifact is bundled, so official signals remain suppressed.

Run the unchanged-gate price-action experiment with:

```text
npm.cmd run research:price-action:windows
```

Run the pooled directional stability and trade-cadence experiment with:

```text
npm.cmd run research:pooled-directional:windows
```

This Step 20 run uses seven purged walk-forward folds and preserves a two-fold
historical diagnostic region. It cannot create a live artifact; a candidate
must still complete a genuinely future shadow period.

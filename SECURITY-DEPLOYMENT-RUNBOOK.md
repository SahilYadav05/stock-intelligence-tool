# Security and Deployment Runbook

## Current truthful deployment state

- The Cloudflare-hosted frontend must be restricted to the operator through Cloudflare Access.
- The Python market-data and ML backend remains a persistent private service; its development
  instance binds to `127.0.0.1:8000`.
- The deployed frontend has no licensed provider or approved model artifacts, so it displays
  `LIVE ANALYSIS UNAVAILABLE` and `WAIT`.
- A hosted HTTPS page must not call an unsecured HTTP backend. In production, the Worker provides
  the same-origin BFF route for only `/api/v1/*` and `/ws/v1/*`; it adds its private bearer
  credential server-to-server and never exposes it to JavaScript.

Cloudflare Workers remains appropriate for the authenticated UI and lightweight edge routes.
The authoritative provider WebSocket, canonical ledger, candle finalization, Python research,
and model inference should run in a persistent backend process. Do not move those responsibilities
to an edge worker merely to claim zero infrastructure cost.

Step 19 price-action responses are computed by that backend from the exact finalized candles in
the requested snapshot. The Cloudflare UI must never recompute or silently substitute levels from
stale browser data. Its conditional setup endpoint remains non-executable even when the live-signal
kill switch is later cleared.

## Development security profile

- `APP_ENV=development`
- backend bound only to `127.0.0.1`
- exact local CORS allowlist
- `API_AUTH_MODE=disabled` is permitted only for the loopback development process
- `LIVE_SIGNAL_KILL_SWITCH=true`
- no provider credentials in the browser, repository, screenshots, logs, or example files

## Production security profile

Startup fails unless all production origins use HTTPS and `API_AUTH_MODE=bearer` has a secret of
at least 32 characters. Bearer mode suits server-to-server research access. A browser deployment
should terminate an HttpOnly authenticated session at a same-origin reverse proxy/BFF; do not put
the backend bearer token in `NEXT_PUBLIC_*`, JavaScript, local storage, or a WebSocket query string.

### Worker gateway configuration

The deployed Worker is deliberately fail-closed. Configure these values in the Cloudflare Worker
dashboard or with Wrangler secrets; do not add them to `.env`, `.env.local`, source control or a
frontend build variable:

```text
MARKET_API_BASE_URL=https://private-market-api.example.com
MARKET_API_BEARER_TOKEN=<the 32+ character backend bearer secret>
```

`MARKET_API_BASE_URL` must be an HTTPS origin for the persistent backend. The Worker passes only
GET/HEAD API and WebSocket requests, strips browser `Authorization` and `Cookie` headers, injects
its own bearer token, and adds `Cache-Control: no-store`. If either secret is absent, clients get
`MARKET_API_PROXY_NOT_CONFIGURED`; no provider response is cached or fabricated.

Protect the Worker route with a Cloudflare Access policy for the operator before setting these
secrets. Without that edge access control, an otherwise public Worker could become a licensed-data
proxy even though the upstream credential remains secret.

For the existing `stock-intelligence.cricwar.workers.dev` Worker, build and publish with:

```text
npm.cmd run build:windows
npx wrangler secret put MARKET_API_BASE_URL --name stock-intelligence
npx wrangler secret put MARKET_API_BEARER_TOKEN --name stock-intelligence
npm.cmd run deploy:workers
```

This requires a valid `wrangler login` for the Cloudflare account that owns the Worker. Keep
`NEXT_PUBLIC_MARKET_API_URL` unset in the production build; it is only the local-development
loopback override.

Required controls:

1. Bind the Python process to a private interface behind an HTTPS proxy.
2. Restrict ingress to the proxy and operator network.
3. Store provider credentials and API secrets in the host secret manager.
4. Use a provider account with no order-placement permission.
5. Keep CORS exact; no wildcard origins.
6. Preserve API no-store and browser security headers.
7. Keep request, rate, and WebSocket connection limits enabled.
8. Back up append-only SQLite ledgers while quiesced or with SQLite's backup API.
9. Verify the audit hash chain after every restore.
10. Rotate secrets after suspected exposure; never log them.

## Release gate

`/api/v1/ready` returns HTTP 503 until every blocker is cleared. A 200 response is possible only
when all of these are simultaneously true:

- licensed provider is configured in live mode;
- canonical snapshot is LIVE;
- chart and analysis use the exact snapshot ID and candle revision checksum;
- model and calibration files are present and their bytes match the SHA-256 manifest;
- feature, label, and signal-policy identities match the running code;
- calibration ECE is at most 0.05 on disjoint chronological evaluation;
- Brier skill is positive;
- reference and current drift distributions each contain at least 100 observations;
- feature PSI is at most 0.20 and probability JSD is at most 0.10;
- the operator kill switch is clear.

Passing this gate authorizes the deterministic signal engine to operate. It does not guarantee a
BUY/SELL; the WAIT policy still applies.

## Model limitation corrected in Step 10

Earlier monitoring could mark drift ready merely because outcome metrics had enough samples. That
was invalid: outcome coverage and distribution drift are different evidence. Drift is now
`UNAVAILABLE` until a versioned reference distribution is supplied and `BREACHED` when a threshold
is exceeded. Either condition blocks live-signal release.

## Storage and recovery

Local MVP ledgers are append-only SQLite with update/delete triggers. Security/release events form
a SHA-256 chain. Before licensed operation:

- place databases outside the source tree under a restricted data directory;
- enable encrypted disk and encrypted backups;
- create a daily backup and a weekly restore drill;
- retain raw provider events only as allowed by the signed data license;
- never back up provider secrets with market data;
- define retention only after the provider confirms storage rights.

For multi-process or multi-user production, replace SQLite with PostgreSQL while preserving the
same append-only contracts. Redis is unnecessary for the one-user MVP; add it only when measured
fan-out or distributed coordination requires it.

## Operational checks

Run before each deployment:

```text
npm.cmd run lint:windows
npm.cmd run test:windows
npm.cmd run benchmark:backend:windows
```

The benchmark measures only in-process API overhead. It explicitly does not measure provider,
internet, candle, model, or browser latency and must not be presented as a trading-latency claim.

After deployment, confirm owner-only access, `/api/v1/live`, blocked readiness without real
evidence, an active cold-start kill switch, absent client-bundle secrets, and matching chart/model
snapshot identities before any signal is shown.

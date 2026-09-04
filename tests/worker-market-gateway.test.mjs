import assert from "node:assert/strict";
import test from "node:test";

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("gateway-test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

function context() {
  return { waitUntil() {}, passThroughOnException() {} };
}

test("market gateway fails closed when its private configuration is absent", async () => {
  const worker = await loadWorker();
  const response = await worker.fetch(
    new Request("https://stock-intelligence.example/api/v1/ready"),
    { ASSETS: { fetch: async () => new Response("not found", { status: 404 }) } },
    context(),
  );

  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    schema_version: 1,
    data_status: "DISCONNECTED",
    reason: "MARKET_API_PROXY_NOT_CONFIGURED",
  });
  assert.equal(response.headers.get("cache-control"), "no-store");
});

test("market gateway forwards only an exact read request with Worker-owned credentials", async () => {
  const worker = await loadWorker();
  const nativeFetch = globalThis.fetch;
  let upstreamUrl;
  let upstreamHeaders;
  globalThis.fetch = async (input, init) => {
    upstreamUrl = String(input);
    upstreamHeaders = new Headers(init?.headers);
    return Response.json({ ok: true }, { headers: { "cache-control": "public, max-age=60" } });
  };

  try {
    const response = await worker.fetch(
      new Request("https://stock-intelligence.example/api/v1/market-state/NIFTY50_SPOT?timeframe=5m", {
        headers: { authorization: "Bearer browser-supplied", cookie: "session=browser" },
      }),
      {
        ASSETS: { fetch: async () => new Response("not found", { status: 404 }) },
        MARKET_API_BASE_URL: "https://private-market.example/ignored-path",
        MARKET_API_BEARER_TOKEN: "worker-only-secret",
      },
      context(),
    );

    assert.equal(response.status, 200);
    assert.equal(upstreamUrl, "https://private-market.example/api/v1/market-state/NIFTY50_SPOT?timeframe=5m");
    assert.equal(upstreamHeaders.get("authorization"), "Bearer worker-only-secret");
    assert.equal(upstreamHeaders.get("cookie"), null);
    assert.equal(upstreamHeaders.get("x-terminal-gateway"), "cloudflare-worker");
    assert.equal(response.headers.get("cache-control"), "no-store");
  } finally {
    globalThis.fetch = nativeFetch;
  }
});

test("market gateway refuses writes before contacting the private backend", async () => {
  const worker = await loadWorker();
  const response = await worker.fetch(
    new Request("https://stock-intelligence.example/api/v1/live", { method: "POST" }),
    {
      ASSETS: { fetch: async () => new Response("not found", { status: 404 }) },
      MARKET_API_BASE_URL: "https://private-market.example",
      MARKET_API_BEARER_TOKEN: "worker-only-secret",
    },
    context(),
  );

  assert.equal(response.status, 405);
  assert.equal((await response.json()).reason, "MARKET_GATEWAY_READ_ONLY");
});

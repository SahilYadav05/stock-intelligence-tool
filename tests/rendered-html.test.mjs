import assert from "node:assert/strict";
import test from "node:test";

async function renderHomePage() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("renders the Step 22 terminal, release gates, and disconnected truth state", async () => {
  const response = await renderHomePage();
  const html = await response.text();

  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  assert.match(html, /NIFTY Intelligence Terminal/i);
  assert.match(html, /Step(?:\s|<!--.*?-->)*22(?:\s|<!--.*?-->)*of(?:\s|<!--.*?-->)*22/i);
  assert.match(html, /Interactive NIFTY 50 candlestick chart/i);
  assert.match(html, /SYNCING ANALYSIS/i);
  assert.match(html, /No prices are being fabricated/i);
  assert.match(html, /price_features\.v1/i);
  assert.match(html, /nifty_5m_atr_first_touch\.v1/i);
  assert.match(html, /Decision engine/i);
  assert.match(html, /Conditional price action/i);
  assert.match(html, /Calibrated probability/i);
  assert.match(html, /Market context/i);
  assert.match(html, /News &amp; events/i);
  assert.match(html, /Historical analogs/i);
  assert.match(html, /Data integrity/i);
  assert.match(html, /No chart data is simulated/i);
  assert.match(html, /--:--:--(?:\s|<!--.*?-->)*IST/i);
  assert.match(html, /Prediction analytics/i);
  assert.match(html, /Paper journal/i);
  assert.match(html, /System monitoring/i);
  assert.match(html, /PAPER ONLY/i);
  assert.match(html, /no cash-performance claim/i);
  assert.match(html, /No paper trades recorded/i);
  assert.match(html, /Live-signal readiness/i);
  assert.match(html, /Signal permission/i);
  assert.match(html, /No gate can be overridden/i);
  assert.doesNotMatch(html, /codex-preview/i);
});

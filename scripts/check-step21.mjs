import { readFileSync } from "node:fs";

const required = [
  "services/backend/src/nifty_terminal/research/step21.py",
  "services/backend/src/nifty_terminal/cli/run_event_price_action_research.py",
  "services/backend/tests/test_step21_event_price_action.py",
  "docs/step21-event-price-action.md",
];

for (const path of required) {
  const text = readFileSync(path, "utf8");
  if (!text.trim()) throw new Error(`Step 21 file is empty: ${path}`);
}

const methodology = readFileSync(required[0], "utf8");
for (const marker of [
  "event_price_action_research.v1",
  "LIQUIDITY_SWEEP_REVERSAL",
  "CONFIRMED_STRUCTURE_BREAK",
  "COMPRESSION_EXPANSION",
  "TREND_EMA_RECLAIM",
  "MAX_TRADES_PER_SESSION = 5",
  "FORWARD_CONFIRMATION_NOT_COMPLETED",
  '"automatic_trading_enabled": False',
]) {
  if (!methodology.includes(marker)) throw new Error(`Step 21 marker missing: ${marker}`);
}

console.log("Step 21 event-based price-action research checks passed.");

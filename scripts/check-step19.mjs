import { readFileSync } from "node:fs";

const required = [
  "services/backend/src/nifty_terminal/price_action/engine.py",
  "services/backend/src/nifty_terminal/price_action/models.py",
  "services/backend/src/nifty_terminal/features/research_v4.py",
  "services/backend/src/nifty_terminal/research/step19.py",
  "services/backend/src/nifty_terminal/cli/run_price_action_research.py",
  "services/backend/tests/test_price_action.py",
  "docs/step19-price-action.md",
  "contracts/price-action-analysis.v1.schema.json",
];

for (const path of required) readFileSync(path);
const engine = readFileSync(required[0], "utf8");
const contract = readFileSync(required[1], "utf8");
const research = readFileSync(required[2], "utf8");
if (!contract.includes('"official_signal": False') || !engine.includes("finalized")) {
  throw new Error("Price-action runtime must remain research-only and finalized-candle based");
}
if (!research.includes("decision_index - 2") || !research.includes("no developing or future candle")) {
  throw new Error("Price-action research must document its causal pivot boundary");
}
console.log(`Step 19 check passed (${required.length} required paths).`);

import { readFileSync } from "node:fs";

const required = [
  "services/backend/src/nifty_terminal/research/step20.py",
  "services/backend/src/nifty_terminal/cli/run_pooled_directional_research.py",
  "services/backend/tests/test_step20_pooled_directional.py",
  "docs/step20-pooled-directional.md",
];

for (const path of required) readFileSync(path);
const research = readFileSync(required[0], "utf8");
for (const text of [
  "n_splits=7",
  "same_timestamp_long_short_rows_kept_in_same_fold",
  "trade_frequency_is_a_hard_gate",
  "CALIBRATION_RANK_REVERSAL",
  '"approved_for_live_inference": False',
  '"automatic_trading_enabled": False',
]) {
  if (!research.includes(text)) throw new Error(`Step 20 invariant missing: ${text}`);
}
console.log(`Step 20 check passed (${required.length} required paths).`);

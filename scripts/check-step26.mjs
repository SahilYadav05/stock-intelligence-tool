import { readFileSync } from "node:fs";

const files = [
  "services/backend/src/nifty_terminal/price_action/replay.py",
  "services/backend/src/nifty_terminal/research/step26.py",
  "services/backend/src/nifty_terminal/cli/run_live_plan_meta_research.py",
  "services/backend/tests/test_price_action_replay.py",
  "services/backend/tests/test_step26_live_plan_meta.py",
  "docs/step26-live-plan-meta.md",
];
for (const path of files) {
  if (!readFileSync(path, "utf8").trim()) {
    throw new Error(`Step 26 file is empty: ${path}`);
  }
}
const replay = readFileSync(files[0], "utf8");
const research = readFileSync(files[1], "utf8");
for (const marker of [
  "SCALE_50_30_20_PROTECTED",
  "STOP_FIRST_CONSERVATIVE",
  "MISSED_GAP_BEYOND_ENTRY_ZONE",
]) {
  if (!replay.includes(marker)) throw new Error(`Step 26 replay marker missing: ${marker}`);
}
for (const marker of [
  "production price-action engine; meta-model cannot reverse it",
  "HISTORICAL_PERIOD_PREVIOUSLY_SEEN",
  "EXECUTABLE_FUTURES_HISTORY_NOT_AVAILABLE",
  "FORWARD_CONFIRMATION_NOT_COMPLETED",
]) {
  if (!research.includes(marker)) throw new Error(`Step 26 research marker missing: ${marker}`);
}
console.log("Step 26 live-plan-aligned meta-label checks passed.");

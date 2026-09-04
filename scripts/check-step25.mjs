import { readFileSync } from "node:fs";

const files = [
  "services/backend/src/nifty_terminal/research/step25.py",
  "services/backend/src/nifty_terminal/cli/run_compact_feature_audit.py",
  "services/backend/tests/test_step25_compact_feature_audit.py",
  "docs/step25-compact-feature-audit.md",
];
for (const path of files) {
  if (!readFileSync(path, "utf8").trim()) {
    throw new Error(`Step 25 file is empty: ${path}`);
  }
}
const source = readFileSync(files[0], "utf8");
for (const marker of [
  "PRICE_ACTION_12",
  "STRUCTURE_LEVELS_COMPACT",
  "LEGACY_STRUCTURE_45",
  "final_historical_folds_reused_for_tuning",
  "DERIVATIVES_FORWARD_DATA_NOT_READY",
]) {
  if (!source.includes(marker)) throw new Error(`Step 25 marker missing: ${marker}`);
}
console.log("Step 25 compact feature-family checks passed.");

import { readFileSync } from "node:fs";

const files = [
  "services/backend/src/nifty_terminal/research/step23.py",
  "services/backend/src/nifty_terminal/cli/run_conditional_direction_research.py",
  "services/backend/tests/test_step23_conditional_direction.py",
  "docs/step23-conditional-direction.md",
];
for (const path of files) {
  if (!readFileSync(path, "utf8").trim()) {
    throw new Error(`Step 23 file is empty: ${path}`);
  }
}
const source = readFileSync(files[0], "utf8");
for (const marker of [
  "ConditionalLabels",
  "opportunity_head",
  "direction_head",
  "diagnostic_thresholds_locked_before_diagnostic",
  "FORWARD_CONFIRMATION_NOT_COMPLETED",
]) {
  if (!source.includes(marker)) throw new Error(`Step 23 marker missing: ${marker}`);
}
console.log("Step 23 conditional-direction checks passed.");

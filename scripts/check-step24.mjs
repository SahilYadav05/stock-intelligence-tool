import { readFileSync } from "node:fs";

const files = [
  "services/backend/src/nifty_terminal/derivatives/features.py",
  "services/backend/src/nifty_terminal/cli/audit_derivatives_readiness.py",
  "services/backend/tests/test_derivatives_features.py",
  "docs/step24-derivatives-features.md",
];
for (const path of files) {
  if (!readFileSync(path, "utf8").trim()) {
    throw new Error(`Step 24 file is empty: ${path}`);
  }
}
const source = readFileSync(files[0], "utf8");
for (const marker of [
  "derivatives__basis_bps",
  "DERIVATIVES_COMPLETE_ROW_SUPPORT_TOO_LOW",
  "OPTIONS_CONTEXT_COMPLETENESS_TOO_LOW",
  "ready_for_model_research",
]) {
  if (!source.includes(marker)) throw new Error(`Step 24 marker missing: ${marker}`);
}
console.log("Step 24 derivatives feature/readiness checks passed.");

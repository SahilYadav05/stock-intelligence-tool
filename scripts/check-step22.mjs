import { readFileSync } from "node:fs";

const files = [
  "services/backend/src/nifty_terminal/derivatives/forward.py",
  "services/backend/src/nifty_terminal/cli/collect_derivatives_snapshot.py",
  "services/backend/tests/test_derivatives_forward.py",
  "docs/step22-derivatives-forward-data.md",
];
for (const path of files) {
  if (!readFileSync(path, "utf8").trim()) throw new Error(`Step 22 file is empty: ${path}`);
}
const source = readFileSync(files[0], "utf8");
for (const marker of ["futures_open_interest", "delta25_put_call_iv_skew", "research_only", "INSERT OR IGNORE"]) {
  if (!source.includes(marker)) throw new Error(`Step 22 marker missing: ${marker}`);
}
const cli = readFileSync(files[1], "utf8");
if (/placeOrder|modifyOrder|cancelOrder/.test(cli)) throw new Error("Step 22 must not expose order methods");
console.log("Step 22 derivatives forward-data checks passed.");

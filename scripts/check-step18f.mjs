import { access, readFile } from "node:fs/promises";

const required = [
  "services/backend/src/nifty_terminal/research/step18f.py",
  "services/backend/src/nifty_terminal/cli/run_baseline_controlled_research.py",
  "services/backend/tests/test_baseline_controlled_research.py",
  "contracts/baseline-controlled-research.v1.schema.json",
  "docs/step18f-baseline-controlled.md",
  "STEP-18F-INSTRUCTIONS.txt"
];
await Promise.all(required.map((path) => access(path)));
const pkg = JSON.parse(await readFile("package.json", "utf8"));
for (const command of ["check:step18f", "research:baseline-controlled:windows"]) {
  if (!pkg.scripts?.[command]) throw new Error(`Missing package script: ${command}`);
}
console.log(`Step 18F check passed (${required.length} required paths).`);

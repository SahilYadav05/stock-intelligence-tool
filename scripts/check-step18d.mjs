import { access, readFile } from "node:fs/promises";

const required = [
  "services/backend/src/nifty_terminal/research/step18d.py",
  "services/backend/src/nifty_terminal/cli/run_selective_utility_research.py",
  "services/backend/tests/test_selective_utility_research.py",
  "contracts/selective-utility-research.v1.schema.json",
  "docs/step18d-selective-utility.md",
  "STEP-18D-INSTRUCTIONS.txt"
];
await Promise.all(required.map((path) => access(path)));
const pkg = JSON.parse(await readFile("package.json", "utf8"));
for (const command of ["check:step18d", "research:selective-utility:windows"]) {
  if (!pkg.scripts?.[command]) throw new Error(`Missing package script: ${command}`);
}
console.log(`Step 18D check passed (${required.length} required paths).`);

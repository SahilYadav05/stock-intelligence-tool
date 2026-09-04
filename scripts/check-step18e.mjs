import { access, readFile } from "node:fs/promises";

const required = [
  "services/backend/src/nifty_terminal/research/step18e.py",
  "services/backend/src/nifty_terminal/cli/run_ranked_utility_research.py",
  "services/backend/tests/test_ranked_utility_research.py",
  "contracts/ranked-utility-research.v1.schema.json",
  "docs/step18e-ranked-utility.md",
  "STEP-18E-INSTRUCTIONS.txt"
];
await Promise.all(required.map((path) => access(path)));
const pkg = JSON.parse(await readFile("package.json", "utf8"));
for (const command of ["check:step18e", "research:ranked-utility:windows"]) {
  if (!pkg.scripts?.[command]) throw new Error(`Missing package script: ${command}`);
}
console.log(`Step 18E check passed (${required.length} required paths).`);

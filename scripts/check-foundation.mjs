import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  ".env.example",
  "app/layout.tsx",
  "app/page.tsx",
  "contracts/README.md",
  "services/backend/pyproject.toml",
  "services/backend/src/nifty_terminal/settings.py",
  "src/config/project.ts",
];

const failures = [];

for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing required path: ${path}`);
  }
}

const page = `${await readFile("app/page.tsx", "utf8")}\n${await readFile("src/components/live-market-terminal.tsx", "utf8")}`;
const forbiddenClaims = ["BUY —", "SELL —", "Probability:", "Accuracy:"];

for (const claim of forbiddenClaims) {
  if (page.includes(claim)) {
    failures.push(`Step 1 UI contains a prohibited unsupported claim: ${claim}`);
  }
}

if (!page.includes("LIVE ANALYSIS UNAVAILABLE")) {
  failures.push("Step 1 UI must state that live analysis is unavailable.");
}

if (failures.length > 0) {
  console.error("Foundation check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Foundation check passed (${requiredPaths.length} required paths).`);

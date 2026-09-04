import { access, readFile } from "node:fs/promises";

const required = [
  "services/backend/src/nifty_terminal/context/__init__.py",
  "services/backend/src/nifty_terminal/context/bundle.py",
  "services/backend/src/nifty_terminal/context/angelone_history.py",
  "services/backend/src/nifty_terminal/context/features.py",
  "services/backend/src/nifty_terminal/research/step18c.py",
  "services/backend/src/nifty_terminal/cli/acquire_angelone_context.py",
  "services/backend/src/nifty_terminal/cli/run_cross_market_research.py",
  "services/backend/tests/test_cross_market_context.py",
  "contracts/cross-market-research.v1.schema.json",
  "docs/step18c-cross-market-research.md"
];
await Promise.all(required.map((path) => access(path)));
const pkg = JSON.parse(await readFile("package.json", "utf8"));
for (const command of [
  "acquire:context:angelone:windows",
  "research:cross-market:windows",
  "check:step18c"
]) {
  if (!pkg.scripts?.[command]) throw new Error(`Missing package script: ${command}`);
}
console.log(`Step 18C check passed (${required.length} required paths).`);

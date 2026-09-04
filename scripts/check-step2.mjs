import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/market-event.v1.schema.json",
  "contracts/provider-health.v1.schema.json",
  "services/backend/src/nifty_terminal/domain/market_event.py",
  "services/backend/src/nifty_terminal/providers/base.py",
  "services/backend/src/nifty_terminal/providers/replay.py",
  "services/backend/src/nifty_terminal/ingestion/normalizer.py",
  "services/backend/src/nifty_terminal/ingestion/validator.py",
  "services/backend/src/nifty_terminal/ingestion/deduplicator.py",
  "services/backend/src/nifty_terminal/ingestion/ledger.py",
  "services/backend/src/nifty_terminal/ingestion/pipeline.py",
  "services/backend/src/nifty_terminal/ingestion/sequence.py",
  "services/backend/tests/fixtures/nifty50_replay.jsonl",
];

const failures = [];

for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 2 path: ${path}`);
  }
}

for (const schemaPath of [
  "contracts/market-event.v1.schema.json",
  "contracts/provider-health.v1.schema.json",
]) {
  try {
    const schema = JSON.parse(await readFile(schemaPath, "utf8"));
    if (!schema.$id || !schema.$id.endsWith(".v1")) {
      failures.push(`Contract is not explicitly versioned: ${schemaPath}`);
    }
  } catch (error) {
    failures.push(`Invalid JSON contract ${schemaPath}: ${error.message}`);
  }
}

const page = `${await readFile("app/page.tsx", "utf8")}\n${await readFile("src/components/live-market-terminal.tsx", "utf8")}`;
if (!page.includes("LIVE ANALYSIS UNAVAILABLE")) {
  failures.push("Step 2 UI must keep live analysis disabled.");
}

const backendFiles = await Promise.all(
  requiredPaths
    .filter((path) => path.endsWith(".py"))
    .map((path) => readFile(path, "utf8")),
);
const backendSource = backendFiles.join("\n").toLowerCase();
for (const providerSdk of ["truedata", "globaldatafeeds", "dhanhq", "upstox", "kiteconnect"]) {
  if (backendSource.includes(`import ${providerSdk}`)) {
    failures.push(`Step 2 must not import an unapproved provider SDK: ${providerSdk}`);
  }
}

if (failures.length > 0) {
  console.error("Step 2 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 2 check passed (${requiredPaths.length} required paths).`);

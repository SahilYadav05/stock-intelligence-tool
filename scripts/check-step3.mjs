import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/candle.v1.schema.json",
  "contracts/market-state-snapshot.v1.schema.json",
  "services/backend/src/nifty_terminal/calendar/nse.py",
  "services/backend/src/nifty_terminal/domain/candle.py",
  "services/backend/src/nifty_terminal/candles/developing.py",
  "services/backend/src/nifty_terminal/candles/engine.py",
  "services/backend/src/nifty_terminal/candles/store.py",
  "services/backend/src/nifty_terminal/snapshots/models.py",
  "services/backend/src/nifty_terminal/snapshots/builder.py",
  "services/backend/src/nifty_terminal/snapshots/store.py",
  "services/backend/tests/test_nse_calendar.py",
  "services/backend/tests/test_developing_candle.py",
  "services/backend/tests/test_candle_engine.py",
  "services/backend/tests/test_snapshots.py",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 3 path: ${path}`);
  }
}

for (const schemaPath of [
  "contracts/candle.v1.schema.json",
  "contracts/market-state-snapshot.v1.schema.json",
]) {
  try {
    const schema = JSON.parse(await readFile(schemaPath, "utf8"));
    if (!schema.$id?.endsWith(".v1") || schema.properties?.schema_version?.const !== 1) {
      failures.push(`Contract is not explicitly versioned: ${schemaPath}`);
    }
  } catch (error) {
    failures.push(`Invalid JSON contract ${schemaPath}: ${error.message}`);
  }
}

const [page, candleSource, snapshotSource] = await Promise.all([
  Promise.all([
    readFile("app/page.tsx", "utf8"),
    readFile("src/components/live-market-terminal.tsx", "utf8"),
  ]).then((parts) => parts.join("\n")),
  readFile("services/backend/src/nifty_terminal/domain/candle.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/snapshots/models.py", "utf8"),
]);

if (!page.includes("LIVE ANALYSIS UNAVAILABLE")) {
  failures.push("Step 3 UI must keep live analysis disabled.");
}
if (!candleSource.includes('DEVELOPING = "DEVELOPING"')) {
  failures.push("Candle domain must distinguish developing candles.");
}
if (!snapshotSource.includes("candle_revision_checksum")) {
  failures.push("Snapshot contract must identify its candle revision set.");
}

if (failures.length > 0) {
  console.error("Step 3 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 3 check passed (${requiredPaths.length} required paths).`);

import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/historical-dataset.v1.schema.json",
  "contracts/feature-snapshot.v1.schema.json",
  "services/backend/src/nifty_terminal/history/models.py",
  "services/backend/src/nifty_terminal/history/quality.py",
  "services/backend/src/nifty_terminal/history/pipeline.py",
  "services/backend/src/nifty_terminal/history/sqlite_repository.py",
  "services/backend/src/nifty_terminal/history/sources/base.py",
  "services/backend/src/nifty_terminal/history/sources/csv_source.py",
  "services/backend/src/nifty_terminal/features/definitions.py",
  "services/backend/src/nifty_terminal/features/engine.py",
  "services/backend/src/nifty_terminal/features/snapshot.py",
  "services/backend/src/nifty_terminal/features/materializer.py",
  "services/backend/src/nifty_terminal/cli/import_history.py",
  "services/backend/tests/test_history_pipeline.py",
  "services/backend/tests/test_features.py",
  "services/backend/tests/test_feature_snapshot.py",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 5 path: ${path}`);
  }
}

for (const schemaPath of [
  "contracts/historical-dataset.v1.schema.json",
  "contracts/feature-snapshot.v1.schema.json",
]) {
  try {
    const schema = JSON.parse(await readFile(schemaPath, "utf8"));
    if (!schema.$id?.endsWith(".v1")) failures.push(`Unversioned contract: ${schemaPath}`);
  } catch (error) {
    failures.push(`Invalid JSON contract ${schemaPath}: ${error.message}`);
  }
}

const [features, repository, terminal] = await Promise.all([
  readFile("services/backend/src/nifty_terminal/features/engine.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/history/sqlite_repository.py", "utf8"),
  readFile("src/components/live-market-terminal.tsx", "utf8"),
]);

if (!features.includes("CandleStatus.FINALIZED")) {
  failures.push("Feature engine must reject developing candles.");
}
if (!features.includes("INTRADAY_GAP_IN_FEATURE_WINDOW")) {
  failures.push("Feature readiness must expose gaps in the active window.");
}
if (!repository.includes("append-only table")) {
  failures.push("Historical research tables must reject update and delete operations.");
}
if (!terminal.includes("price_features.v1") && !terminal.includes("PROJECT.featureVersion")) {
  failures.push("Terminal must expose the feature version without invented values.");
}
if (!terminal.includes("LIVE ANALYSIS UNAVAILABLE")) {
  failures.push("Step 5 must not claim live analysis or a model signal.");
}

if (failures.length) {
  console.error("Step 5 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 5 check passed (${requiredPaths.length} required paths).`);

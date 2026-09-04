import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/tracked-prediction.v1.schema.json",
  "contracts/prediction-assessment.v1.schema.json",
  "contracts/paper-trade.v1.schema.json",
  "contracts/paper-trade-event.v1.schema.json",
  "contracts/monitoring-view.v1.schema.json",
  "contracts/tracking-overview.v1.schema.json",
  "services/backend/src/nifty_terminal/tracking/models.py",
  "services/backend/src/nifty_terminal/tracking/paper.py",
  "services/backend/src/nifty_terminal/tracking/analytics.py",
  "services/backend/src/nifty_terminal/tracking/monitoring.py",
  "services/backend/src/nifty_terminal/tracking/read_model.py",
  "services/backend/src/nifty_terminal/tracking/service.py",
  "services/backend/src/nifty_terminal/tracking/sqlite_repository.py",
  "services/backend/tests/test_tracking.py",
  "src/lib/tracking-contracts.ts",
  "src/lib/tracking-client.ts",
  "src/components/tracking-console.tsx",
  "TRACKING-AND-MONITORING-CONTRACT.md",
  "STEP-9-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 9 path: ${path}`);
  }
}

for (const schemaPath of requiredPaths.filter((path) => path.startsWith("contracts/"))) {
  try {
    const schema = JSON.parse(await readFile(schemaPath, "utf8"));
    if (!schema.$id?.endsWith(".v1")) failures.push(`Unversioned contract: ${schemaPath}`);
  } catch (error) {
    failures.push(`Invalid JSON contract ${schemaPath}: ${error.message}`);
  }
}

const [api, models, paper, analytics, ledger, client, consoleSource] = await Promise.all([
  readFile("services/backend/src/nifty_terminal/api/app.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/tracking/models.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/tracking/paper.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/tracking/analytics.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/tracking/sqlite_repository.py", "utf8"),
  readFile("src/lib/tracking-contracts.ts", "utf8"),
  readFile("src/components/tracking-console.tsx", "utf8"),
]);

if (!api.includes('/api/v1/tracking/{instrument_id}') || !api.includes("minimum_analytics_sample")) {
  failures.push("Step 9 tracking and monitoring API is missing.");
}
if (!models.includes('"automatic_execution": False') || !models.includes('"performance_claim_allowed": False')) {
  failures.push("Paper execution and performance claims must remain disabled.");
}
if (!paper.includes("AMBIGUOUS_INTRABAR_STOP_AND_TARGET_ORDER") || !models.includes("WAIT cannot create")) {
  failures.push("Paper engine must fail safely on WAIT and ambiguous intrabar order.");
}
if (!analytics.includes("MINIMUM_REPORTING_SAMPLE = 30") || !analytics.includes("INSUFFICIENT_SAMPLE")) {
  failures.push("Analytics must suppress precise metrics before the sample gate.");
}
if (!ledger.includes("append-only table") || !ledger.includes("_no_update") || !ledger.includes("_no_delete")) {
  failures.push("Tracking storage must be append-only.");
}
if (!client.includes("paper_only") || !client.includes("automatic_execution")) {
  failures.push("Browser must reject non-paper or execution-capable tracking payloads.");
}
for (const section of ["Prediction analytics", "Paper journal", "System monitoring", "no cash-performance claim"]) {
  if (!consoleSource.includes(section)) failures.push(`Tracking dashboard section missing: ${section}`);
}

if (failures.length) {
  console.error("Step 9 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 9 check passed (${requiredPaths.length} required paths).`);

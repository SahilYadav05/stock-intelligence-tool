import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/analysis-view.v1.schema.json",
  "contracts/analysis-availability.v1.schema.json",
  "services/backend/src/nifty_terminal/dashboard/models.py",
  "services/backend/src/nifty_terminal/dashboard/read_model.py",
  "services/backend/tests/dashboard_fixture.py",
  "services/backend/tests/test_dashboard.py",
  "src/lib/analysis-contracts.ts",
  "src/lib/analysis-client.ts",
  "src/components/terminal-chart.tsx",
  "src/components/live-market-terminal.tsx",
  "DASHBOARD-UX-CONTRACT.md",
  "STEP-8-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 8 path: ${path}`);
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

const [api, dashboard, client, chart, terminal] = await Promise.all([
  readFile("services/backend/src/nifty_terminal/api/app.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/dashboard/models.py", "utf8"),
  readFile("src/lib/analysis-contracts.ts", "utf8"),
  readFile("src/components/terminal-chart.tsx", "utf8"),
  readFile("src/components/live-market-terminal.tsx", "utf8"),
]);

if (!api.includes('/api/v1/analysis/{instrument_id}') || !api.includes('ANALYSIS_REVISION_MISMATCH')) {
  failures.push("Backend must deliver analysis for one exact market snapshot revision.");
}
if (!dashboard.includes("Signal and analysis must reference the same snapshot") || !dashboard.includes("candle revisions must match")) {
  failures.push("Analysis construction must reject signal/snapshot revision mismatch.");
}
if (!client.includes("expectedSnapshotId") || !client.includes("expectedRevisionChecksum")) {
  failures.push("Browser analysis parsing must verify snapshot and revision identity.");
}
for (const overlay of ["ENTRY LOW", "STOP", "TARGET 1", "SUPPORT", "RESISTANCE"]) {
  if (!chart.includes(overlay)) failures.push(`Chart overlay missing: ${overlay}`);
}
for (const section of ["Decision engine", "Market context", "News & events", "Historical analogs", "Data integrity"]) {
  if (!terminal.includes(section)) failures.push(`Professional dashboard section missing: ${section}`);
}
if (!terminal.includes("No chart data is simulated") || !terminal.includes("LIVE ANALYSIS UNAVAILABLE")) {
  failures.push("Dashboard must preserve honest disconnected and no-fabrication states.");
}
if (terminal.includes("formatIstWithSeconds(new Date().toISOString())")) {
  failures.push("Clock rendering must not read the current time during server/client hydration.");
}
if (!terminal.includes("const [clockIso, setClockIso] = useState<string | null>(null)")) {
  failures.push("Dashboard clock must use a hydration-stable initial state.");
}

if (failures.length) {
  console.error("Step 8 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 8 check passed (${requiredPaths.length} required paths).`);

import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/real-data-research.v1.schema.json",
  "services/backend/src/nifty_terminal/research/__init__.py",
  "services/backend/src/nifty_terminal/research/real_data.py",
  "services/backend/src/nifty_terminal/cli/run_real_data_research.py",
  "services/backend/src/nifty_terminal/history/sqlite_repository.py",
  "services/backend/tests/test_real_data_research.py",
  "STEP-14-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 14 path: ${path}`);
  }
}

const [contractSource, research, cli, repository, packageSource] = await Promise.all([
  readFile("contracts/real-data-research.v1.schema.json", "utf8"),
  readFile("services/backend/src/nifty_terminal/research/real_data.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/cli/run_real_data_research.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/history/sqlite_repository.py", "utf8"),
  readFile("package.json", "utf8"),
]);

const contract = JSON.parse(contractSource);
if (contract.properties?.approved_for_live_inference?.const !== false) {
  failures.push("The Step 14 contract must prohibit live-inference approval.");
}

for (const requirement of [
  "minimum_oos_predictions\": 7_500",
  "minimum_folds\": 5",
  "CALIBRATION_RELEASE_GATE_NOT_PASSED",
  '"news_used": False',
  '"approved_for_live_inference": False',
]) {
  if (!research.includes(requirement)) {
    failures.push(`Real-data research safety requirement missing: ${requirement}`);
  }
}
for (const requirement of [
  'DEFAULT_CALENDAR = Path("config/nse-calendar-through-2026-08-25.json")',
  "default=DEFAULT_CALENDAR",
  "Sourced exchange calendar does not exist",
  "LIVE_SIGNAL_KILL_SWITCH=true",
  "REAL_DATA_WALK_FORWARD_CONFIG = WalkForwardConfig(",
  "n_splits=5",
  "test_samples=2_000",
  "minimum_train_samples=10_000",
  "minimum_train_class_samples=25",
  "do not tune thresholds to force a pass",
]) {
  if (!cli.includes(requirement)) {
    failures.push(`Step 14 CLI requirement missing: ${requirement}`);
  }
}
if (!repository.includes("def latest_pass_dataset_id")) {
  failures.push("The repository cannot select the latest immutable PASS dataset.");
}

const packageJson = JSON.parse(packageSource);
for (const command of ["check:step14", "research:real-data", "research:real-data:windows"]) {
  if (!packageJson.scripts[command]) failures.push(`Missing package script: ${command}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step14")) {
    failures.push(`${command} does not include the Step 14 structural gate.`);
  }
}

if (failures.length) {
  console.error("Step 14 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 14 check passed (${requiredPaths.length} required paths).`);

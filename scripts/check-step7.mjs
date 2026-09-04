import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/calibration-run.v1.schema.json",
  "contracts/calibrated-prediction.v1.schema.json",
  "contracts/signal-decision.v1.schema.json",
  "contracts/signal-lifecycle-event.v1.schema.json",
  "services/backend/src/nifty_terminal/calibration/definitions.py",
  "services/backend/src/nifty_terminal/calibration/models.py",
  "services/backend/src/nifty_terminal/calibration/temperature.py",
  "services/backend/src/nifty_terminal/calibration/pipeline.py",
  "services/backend/src/nifty_terminal/calibration/research.py",
  "services/backend/src/nifty_terminal/signals/definitions.py",
  "services/backend/src/nifty_terminal/signals/models.py",
  "services/backend/src/nifty_terminal/signals/policy.py",
  "services/backend/src/nifty_terminal/signals/lifecycle.py",
  "services/backend/src/nifty_terminal/signals/replay.py",
  "services/backend/src/nifty_terminal/cli/calibrate_policy.py",
  "services/backend/tests/test_calibration.py",
  "services/backend/tests/test_signal_policy.py",
  "services/backend/tests/test_step7_storage.py",
  "CALIBRATION-SIGNAL-METHODOLOGY.md",
  "STEP-7-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 7 path: ${path}`);
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

const [calibration, policy, lifecycle, repository, terminal] = await Promise.all([
  readFile("services/backend/src/nifty_terminal/calibration/pipeline.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/signals/policy.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/signals/lifecycle.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/history/sqlite_repository.py", "utf8"),
  readFile("src/components/live-market-terminal.tsx", "utf8"),
]);

if (!calibration.includes("fit_rows, evaluation_rows") || !calibration.includes("BRIER_SKILL_RELEASE_GATE_FAILED")) {
  failures.push("Calibration must use disjoint chronological partitions and a Brier-skill gate.");
}
if (!calibration.includes("ECE_RELEASE_GATE_FAILED") || !calibration.includes("PROBABILITY_BIN")) {
  failures.push("Calibration must gate ECE and displayed probability-bin support.");
}
if (!calibration.includes("CALIBRATION_DEGRADATION_GATE_FAILED")) {
  failures.push("Calibration must not be released when proper scores degrade.");
}
if (!policy.includes("SignalDirection.WAIT") || !policy.includes("DATA_NOT_LIVE")) {
  failures.push("The deterministic signal policy must fail closed to WAIT.");
}
if (!policy.includes("CONFLICTING_ACTIVE_SIGNAL_MUST_INVALIDATE_FIRST")) {
  failures.push("Direct BUY/SELL flips must be blocked by hysteresis.");
}
if (!lifecycle.includes("AMBIGUOUS_INTRABAR_STOP_AND_TARGET_ORDER")) {
  failures.push("Ambiguous lifecycle outcomes must not be guessed.");
}
for (const table of ["calibration_runs", "calibrated_predictions", "signal_decisions", "signal_lifecycle_events"]) {
  if (!repository.includes(`CREATE TABLE IF NOT EXISTS ${table}`)) {
    failures.push(`Missing append-only Step 7 table: ${table}`);
  }
}
if (!terminal.includes("NO APPROVED CALIBRATION") || !terminal.includes("LIVE ANALYSIS UNAVAILABLE")) {
  failures.push("The terminal must expose the current unapproved/disconnected truth state.");
}

if (failures.length) {
  console.error("Step 7 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 7 check passed (${requiredPaths.length} required paths).`);

import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/first-touch-label.v1.schema.json",
  "contracts/ml-training-run.v1.schema.json",
  "contracts/replay-prediction.v1.schema.json",
  "contracts/replay-assessment.v1.schema.json",
  "services/backend/src/nifty_terminal/ml/definitions.py",
  "services/backend/src/nifty_terminal/ml/models.py",
  "services/backend/src/nifty_terminal/ml/labels.py",
  "services/backend/src/nifty_terminal/ml/dataset.py",
  "services/backend/src/nifty_terminal/ml/split.py",
  "services/backend/src/nifty_terminal/ml/metrics.py",
  "services/backend/src/nifty_terminal/ml/training.py",
  "services/backend/src/nifty_terminal/ml/replay.py",
  "services/backend/src/nifty_terminal/ml/pipeline.py",
  "services/backend/src/nifty_terminal/cli/train_replay.py",
  "services/backend/tests/test_ml_dataset.py",
  "services/backend/tests/test_ml_labels.py",
  "services/backend/tests/test_ml_training.py",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 6 path: ${path}`);
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

const [definitions, labels, splitter, replay, terminal] = await Promise.all([
  readFile("services/backend/src/nifty_terminal/ml/definitions.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/ml/labels.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/ml/split.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/ml/replay.py", "utf8"),
  readFile("src/components/live-market-terminal.tsx", "utf8"),
]);

if (!definitions.includes('HORIZON_BARS = 12')) {
  failures.push("The locked target must retain a 12-bar/60-minute horizon.");
}
if (!definitions.includes('UP_ATR_MULTIPLIER = Decimal("1.0")') ||
    !definitions.includes('DOWN_ATR_MULTIPLIER = Decimal("1.0")')) {
  failures.push("The locked target must retain symmetric ±1.0 ATR barriers.");
}
if (!labels.includes("AMBIGUOUS_INTRABAR_ORDER")) {
  failures.push("Unresolved double touches must remain explicitly ambiguous.");
}
if (!splitter.includes("label_window_end <= embargo_cutoff")) {
  failures.push("Chronological folds must purge overlapping label windows.");
}
if (!replay.includes("ReplayAssessment")) {
  failures.push("Replay outcomes must be stored separately from predictions.");
}
if (!terminal.includes("No calibrated probability")) {
  failures.push("Step 6 must not display raw ML output as a calibrated probability.");
}
if (!terminal.includes("NO SIGNAL")) {
  failures.push("Step 6 must not produce an official signal.");
}

if (failures.length) {
  console.error("Step 6 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 6 check passed (${requiredPaths.length} required paths).`);

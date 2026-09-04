import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/locked-shadow-research.v1.schema.json",
  "contracts/shadow-model-artifact.v1.schema.json",
  "services/backend/src/nifty_terminal/research/step16.py",
  "services/backend/src/nifty_terminal/cli/run_locked_shadow_research.py",
  "services/backend/tests/test_locked_shadow_research.py",
  "STEP-16-COMMAND-PROMPT.md",
];
const failures = [];
for (const path of requiredPaths) {
  try { await access(path); } catch { failures.push(`Missing Step 16 path: ${path}`); }
}

const [reportContractSource, artifactContractSource, research, cli, step15, packageSource] =
  await Promise.all([
    readFile("contracts/locked-shadow-research.v1.schema.json", "utf8"),
    readFile("contracts/shadow-model-artifact.v1.schema.json", "utf8"),
    readFile("services/backend/src/nifty_terminal/research/step16.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/cli/run_locked_shadow_research.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/research/v2.py", "utf8"),
    readFile("package.json", "utf8"),
  ]);
const reportContract = JSON.parse(reportContractSource);
const artifactContract = JSON.parse(artifactContractSource);
if (reportContract.properties?.approved_for_live_inference?.const !== false) {
  failures.push("Step 16 report contract must prohibit live inference.");
}
if (reportContract.properties?.precise_probability_display_allowed?.const !== false) {
  failures.push("Step 16 report must prohibit precise live probability display.");
}
if (artifactContract.properties?.shadow_only?.const !== true) {
  failures.push("Step 16 artifact must be shadow-only.");
}
if (artifactContract.properties?.approved_for_live_inference?.const !== false) {
  failures.push("Step 16 artifact must prohibit live inference approval.");
}
for (const requirement of [
  'LOCKED_ATR_MULTIPLIER = Decimal("1.5")',
  'LOCKED_CANDIDATE = "multinomial_logistic_unweighted"',
  'CALIBRATION_METHODS = ("identity", "temperature", "prior_shrinkage", "vector_scaling")',
  '"same_minute_stop_and_target": "STOP_FIRST_CONSERVATIVE"',
  '"one_way_slippage_points": 0.5',
  '"forward_confirmation_required": True',
  '"approved_for_live_inference": False',
]) {
  if (!research.includes(requirement)) failures.push(`Step 16 requirement missing: ${requirement}`);
}
for (const requirement of [
  "_require_live_signal_kill_switch",
  "SAFE_JSON_PARAMETERS_ONLY",
  "No model was released for official live inference or order execution.",
]) {
  if (!cli.includes(requirement) && !research.includes(requirement)) {
    failures.push(`Step 16 CLI/artifact requirement missing: ${requirement}`);
  }
}
if (!step15.includes('"ranking_metric": "BRIER_SKILL_VS_TARGET_SPECIFIC_PRIOR"')) {
  failures.push("Step 15 target-ranking correction is missing.");
}
const packageJson = JSON.parse(packageSource);
for (const command of [
  "check:step16",
  "research:locked-shadow",
  "research:locked-shadow:windows",
]) {
  if (!packageJson.scripts[command]) failures.push(`Missing package script: ${command}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step16")) {
    failures.push(`${command} does not include check:step16.`);
  }
}
if (failures.length) {
  console.error("Step 16 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}
console.log(`Step 16 check passed (${requiredPaths.length} required paths).`);

import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/probability-research-v2.v1.schema.json",
  "services/backend/src/nifty_terminal/research/v2.py",
  "services/backend/src/nifty_terminal/cli/run_probability_research_v2.py",
  "services/backend/tests/test_probability_research_v2.py",
  "STEP-15-COMMAND-PROMPT.md",
];
const failures = [];
for (const path of requiredPaths) {
  try { await access(path); } catch { failures.push(`Missing Step 15 path: ${path}`); }
}
const [contractSource, research, cli, packageSource] = await Promise.all([
  readFile("contracts/probability-research-v2.v1.schema.json", "utf8"),
  readFile("services/backend/src/nifty_terminal/research/v2.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/cli/run_probability_research_v2.py", "utf8"),
  readFile("package.json", "utf8"),
]);
const contract = JSON.parse(contractSource);
if (contract.properties?.approved_for_live_inference?.const !== false) {
  failures.push("Step 15 contract must prohibit live-inference approval.");
}
for (const requirement of [
  'BARRIER_MULTIPLIERS = (Decimal("1.0"), Decimal("1.25"), Decimal("1.5"))',
  '"multinomial_logistic_unweighted"',
  '"hist_gradient_boosting_unweighted"',
  '"candidate_selection_folds": [0, 1, 2]',
  '"calibration_fit_fold": 3',
  '"final_screening_fold": 4',
  '"approved_for_live_inference": False',
]) {
  if (!research.includes(requirement)) failures.push(`Research-v2 requirement missing: ${requirement}`);
}
for (const requirement of [
  "_require_live_signal_kill_switch",
  "symmetric_first_touch_config",
  "No model, precise live probability, official signal or order was released.",
]) {
  if (!cli.includes(requirement)) failures.push(`Step 15 CLI requirement missing: ${requirement}`);
}
const packageJson = JSON.parse(packageSource);
for (const command of ["check:step15", "research:probability-v2", "research:probability-v2:windows"]) {
  if (!packageJson.scripts[command]) failures.push(`Missing package script: ${command}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step15")) {
    failures.push(`${command} does not include check:step15.`);
  }
}
if (failures.length) {
  console.error("Step 15 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}
console.log(`Step 15 check passed (${requiredPaths.length} required paths).`);

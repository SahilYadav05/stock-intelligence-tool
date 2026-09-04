import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/model-v2-research.v1.schema.json",
  "services/backend/src/nifty_terminal/features/enhanced.py",
  "services/backend/src/nifty_terminal/research/step18.py",
  "services/backend/src/nifty_terminal/cli/run_model_v2_research.py",
  "services/backend/tests/test_model_v2_research.py",
  "STEP-18-COMMAND-PROMPT.md",
];
const failures = [];
for (const path of requiredPaths) {
  try { await access(path); } catch { failures.push(`Missing Step 18 path: ${path}`); }
}

const [schemaSource, enhanced, research, cli, packageSource] = await Promise.all([
  readFile("contracts/model-v2-research.v1.schema.json", "utf8"),
  readFile("services/backend/src/nifty_terminal/features/enhanced.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/research/step18.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/cli/run_model_v2_research.py", "utf8"),
  readFile("package.json", "utf8"),
]);
const schema = JSON.parse(schemaSource);
for (const property of [
  "existing_step17_runtime_modified",
  "model_artifact_created",
  "approved_for_live_inference",
  "precise_probability_display_allowed",
  "official_signal_available",
  "automatic_trading_enabled",
]) {
  if (schema.properties?.[property]?.const !== false) {
    failures.push(`Step 18 contract must force ${property}=false.`);
  }
}
for (const requirement of [
  'ENHANCED_FEATURE_VERSION = "price_features.v2"',
  "FINALIZED_5M_15M_1H_ONLY",
  "enhance_sample",
]) {
  if (!enhanced.includes(requirement) && !research.includes(requirement)) {
    failures.push(`Enhanced feature requirement missing: ${requirement}`);
  }
}
for (const requirement of [
  'STEP18_VERSION = "hierarchical_model_v2_research.v1"',
  "hierarchical_logistic_v2",
  "hierarchical_hgb_balanced_v2",
  "combine_hierarchical_probabilities",
  "directional_collapse_detected",
  '"existing_step17_runtime_modified": False',
  '"model_artifact_created": False',
  '"approved_for_live_inference": False',
]) {
  if (!research.includes(requirement)) failures.push(`Step 18 research requirement missing: ${requirement}`);
}
if (!cli.includes("Step 17 shadow manifest and ledger will not be modified")) {
  failures.push("Step 18 CLI does not state Step 17 runtime isolation.");
}
const packageJson = JSON.parse(packageSource);
for (const command of ["check:step18", "research:model-v2", "research:model-v2:windows"]) {
  if (!packageJson.scripts[command]) failures.push(`Missing package script: ${command}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step18")) {
    failures.push(`${command} does not include check:step18.`);
  }
}
if (failures.length) {
  console.error("Step 18 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}
console.log(`Step 18 check passed (${requiredPaths.length} required paths).`);

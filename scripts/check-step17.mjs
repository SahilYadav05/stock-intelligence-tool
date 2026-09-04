import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/shadow-policy-research.v1.schema.json",
  "contracts/shadow-runtime-manifest.v1.schema.json",
  "contracts/shadow-prediction.v1.schema.json",
  "services/backend/src/nifty_terminal/research/step17.py",
  "services/backend/src/nifty_terminal/cli/run_shadow_policy_research.py",
  "services/backend/src/nifty_terminal/cli/verify_shadow_runtime.py",
  "services/backend/src/nifty_terminal/shadow/artifacts.py",
  "services/backend/src/nifty_terminal/shadow/ledger.py",
  "services/backend/src/nifty_terminal/shadow/runtime.py",
  "services/backend/tests/test_shadow_runtime.py",
  "STEP-17-COMMAND-PROMPT.md",
];
const failures = [];
for (const path of requiredPaths) {
  try { await access(path); } catch { failures.push(`Missing Step 17 path: ${path}`); }
}

const [researchContractSource, manifestContractSource, predictionContractSource, research, runtime, liveRuntime, settings, envExample, packageSource] = await Promise.all([
  readFile("contracts/shadow-policy-research.v1.schema.json", "utf8"),
  readFile("contracts/shadow-runtime-manifest.v1.schema.json", "utf8"),
  readFile("contracts/shadow-prediction.v1.schema.json", "utf8"),
  readFile("services/backend/src/nifty_terminal/research/step17.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/shadow/runtime.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/runtime/live_market.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/settings.py", "utf8"),
  readFile(".env.example", "utf8"),
  readFile("package.json", "utf8"),
]);
for (const [name, source] of [
  ["research", researchContractSource],
  ["manifest", manifestContractSource],
  ["prediction", predictionContractSource],
]) {
  const schema = JSON.parse(source);
  if (schema.properties?.official_signal_available?.const !== false && name !== "prediction") {
    failures.push(`${name} contract must prohibit official signals.`);
  }
  if (schema.properties?.automatic_trading_enabled?.const !== false) {
    failures.push(`${name} contract must prohibit automatic trading.`);
  }
}
for (const requirement of [
  'STEP17_VERSION = "shadow_policy_research.v1"',
  '"RAW_MODEL_SCORE"',
  '"CALIBRATED_PROBABILITY"',
  '"FORWARD_CONFIRMATION_NOT_COMPLETED"',
  '"approved_for_live_inference": False',
]) {
  if (!research.includes(requirement)) failures.push(`Step 17 research requirement missing: ${requirement}`);
}
for (const requirement of [
  "official_signal\": None",
  "precise_probability_display_allowed\": False",
  "automatic_trading_enabled\": False",
  "SQLiteShadowLedger",
]) {
  if (!runtime.includes(requirement)) failures.push(`Shadow runtime requirement missing: ${requirement}`);
}
if (!liveRuntime.includes("self._shadow_runtime.process")) {
  failures.push("Finalized live snapshots are not connected to shadow processing.");
}
for (const requirement of ["shadow_mode_enabled", "shadow_runtime_manifest_path", "shadow_ledger_path"]) {
  if (!settings.includes(requirement)) failures.push(`Settings requirement missing: ${requirement}`);
}
for (const requirement of ["SHADOW_MODE_ENABLED=false", "SHADOW_RUNTIME_MANIFEST_PATH=", "SHADOW_LEDGER_PATH="]) {
  if (!envExample.includes(requirement)) failures.push(`.env.example requirement missing: ${requirement}`);
}
const packageJson = JSON.parse(packageSource);
for (const command of [
  "check:step17",
  "research:shadow-policy",
  "research:shadow-policy:windows",
  "verify:shadow-runtime",
  "verify:shadow-runtime:windows",
]) {
  if (!packageJson.scripts[command]) failures.push(`Missing package script: ${command}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step17")) {
    failures.push(`${command} does not include check:step17.`);
  }
}
if (failures.length) {
  console.error("Step 17 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}
console.log(`Step 17 check passed (${requiredPaths.length} required paths).`);

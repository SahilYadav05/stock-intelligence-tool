import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/trade-aligned-research.v1.schema.json",
  "services/backend/src/nifty_terminal/features/research_v3.py",
  "services/backend/src/nifty_terminal/research/step18b.py",
  "services/backend/src/nifty_terminal/cli/run_trade_aligned_research.py",
  "services/backend/tests/test_trade_aligned_research.py",
  "STEP-18B-COMMAND-PROMPT.md",
];
const failures = [];
for (const path of requiredPaths) {
  try { await access(path); } catch { failures.push(`Missing Step 18B path: ${path}`); }
}
const [schemaSource, features, research, cli, packageSource] = await Promise.all([
  readFile("contracts/trade-aligned-research.v1.schema.json", "utf8"),
  readFile("services/backend/src/nifty_terminal/features/research_v3.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/research/step18b.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/cli/run_trade_aligned_research.py", "utf8"),
  readFile("package.json", "utf8"),
]);
const schema = JSON.parse(schemaSource);
for (const property of [
  "existing_step17_runtime_modified",
  "existing_step18_report_modified",
  "model_artifact_created",
  "approved_for_live_inference",
  "precise_probability_display_allowed",
  "official_signal_available",
  "automatic_trading_enabled",
]) {
  if (schema.properties?.[property]?.const !== false) {
    failures.push(`Step 18B contract must force ${property}=false.`);
  }
}
for (const requirement of [
  'RESEARCH_FEATURE_VERSION = "stationary_price_features.v3"',
  "macd_histogram_atr",
  "adx_14",
  "opening_range_position",
  "previous_high_distance_atr",
  "bullish_engulfing",
  "Developing candles cannot enter research features",
]) {
  if (!features.includes(requirement)) failures.push(`Feature-v3 requirement missing: ${requirement}`);
}
for (const requirement of [
  'STEP18B_VERSION = "trade_aligned_model_research.v1.1"',
  'TARGET_ATR = Decimal("1.0")',
  'STOP_ATR = Decimal("0.75")',
  "block_bootstrap_brier_skill",
  "simulate_directional_policy",
  "technical_logistic_l2",
  "stationary_elasticnet",
  "stationary_hgb",
  "MODEL_OPTIMIZER_DID_NOT_CONVERGE",
  "_fit_with_convergence_check",
  '"model_artifact_created": False',
  '"approved_for_live_inference": False',
]) {
  if (!research.includes(requirement)) failures.push(`Step 18B research requirement missing: ${requirement}`);
}
if (!cli.includes("existing Step 17/18 artifacts and ledgers will not be modified")) {
  failures.push("Step 18B CLI does not state artifact isolation.");
}
const packageJson = JSON.parse(packageSource);
for (const command of ["check:step18b", "research:trade-aligned", "research:trade-aligned:windows"]) {
  if (!packageJson.scripts[command]) failures.push(`Missing package script: ${command}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step18b")) {
    failures.push(`${command} does not include check:step18b.`);
  }
}
if (failures.length) {
  console.error("Step 18B check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}
console.log(`Step 18B check passed (${requiredPaths.length} required paths).`);

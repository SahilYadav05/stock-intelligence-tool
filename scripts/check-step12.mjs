import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "services/backend/src/nifty_terminal/runtime/__init__.py",
  "services/backend/src/nifty_terminal/runtime/live_market.py",
  "services/backend/tests/test_live_runtime.py",
  "scripts/check-step12.mjs",
  "STEP-12-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 12 path: ${path}`);
  }
}

const [runtime, app, snapshots, provider, terminal, settings, env, packageSource] =
  await Promise.all([
    readFile("services/backend/src/nifty_terminal/runtime/live_market.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/api/app.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/snapshots/builder.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/providers/angelone.py", "utf8"),
    readFile("src/components/live-market-terminal.tsx", "utf8"),
    readFile("services/backend/src/nifty_terminal/settings.py", "utf8"),
    readFile(".env.example", "utf8"),
    readFile("package.json", "utf8"),
  ]);

for (const requirement of [
  "LiveMarketRuntime",
  "fetch_finalized_minutes",
  "DevelopingCandleEngine",
  "MarketStateSnapshotBuilder",
  "MarketStateDeliveryService",
  "_official_model_signature",
  "developing_candle=developing",
  "persist=False",
]) {
  if (!runtime.includes(requirement)) {
    failures.push(`Continuous runtime requirement missing: ${requirement}`);
  }
}

for (const requirement of [
  "build_angelone_live_runtime",
  "lifespan=lifespan",
  "/api/v1/provider/health",
  "application.state.live_runtime",
]) {
  if (!app.includes(requirement)) {
    failures.push(`Backend lifecycle requirement missing: ${requirement}`);
  }
}

if (!snapshots.includes("data_as_of") || !snapshots.includes("additional_blockers")) {
  failures.push("Snapshot builder is missing point-in-time live metadata.");
}
if (!provider.includes("DISCONNECT_FLAG") || !provider.includes("RESUBSCRIBE_FLAG")) {
  failures.push("Angel One intentional shutdown protection is missing.");
}
if (!terminal.includes("Current price") || !terminal.includes("developing={developing}")) {
  failures.push("The live chart is not bound to the visual developing candle.");
}

for (const variable of [
  "LIVE_HISTORY_LOOKBACK_DAYS",
  "LIVE_HISTORY_RECOVERY_MINUTES",
  "LIVE_HISTORY_POLL_SECONDS",
  "LIVE_MINUTE_FINALIZATION_DELAY_SECONDS",
  "LIVE_TICK_FRESH_SECONDS",
  "LIVE_TICK_STALE_SECONDS",
  "LIVE_CHART_PUBLISH_INTERVAL_MILLISECONDS",
]) {
  if (!settings.includes(variable) || !env.includes(variable)) {
    failures.push(`Live runtime setting missing: ${variable}`);
  }
}

const packageJson = JSON.parse(packageSource);
if (!packageJson.scripts["check:step12"]) {
  failures.push("Missing package script: check:step12");
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step12")) {
    failures.push(`${command} does not include the Step 12 gate.`);
  }
}

if (failures.length) {
  console.error("Step 12 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 12 check passed (${requiredPaths.length} required paths).`);

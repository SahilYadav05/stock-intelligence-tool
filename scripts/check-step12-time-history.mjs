import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "src/lib/chart-history.ts",
  "src/components/terminal-chart.tsx",
  "services/backend/src/nifty_terminal/runtime/live_market.py",
  "scripts/check-step12-time-history.mjs",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing time/history fix path: ${path}`);
  }
}

const [chart, client, terminal, runtime, app, settings, env, packageSource] =
  await Promise.all([
    readFile("src/components/terminal-chart.tsx", "utf8"),
    readFile("src/lib/chart-history.ts", "utf8"),
    readFile("src/components/live-market-terminal.tsx", "utf8"),
    readFile("services/backend/src/nifty_terminal/runtime/live_market.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/api/app.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/settings.py", "utf8"),
    readFile(".env.example", "utf8"),
    readFile("package.json", "utf8"),
  ]);

for (const requirement of [
  'timeZone: "Asia/Kolkata"',
  "tickMarkFormatter",
  "formatIstCrosshairTime",
  "setVisibleLogicalRange",
]) {
  if (!chart.includes(requirement)) failures.push(`IST chart requirement missing: ${requirement}`);
}
for (const requirement of ["fetchChartHistory", "mergeCanonicalCandles"]) {
  if (!client.includes(requirement) || !terminal.includes(requirement)) {
    failures.push(`Chart-history client requirement missing: ${requirement}`);
  }
}
for (const requirement of [
  "chart_history_primary_limit",
  "chart_history_context_limit",
  "chart_history_hourly_limit",
  "def chart_history",
]) {
  if (!runtime.includes(requirement)) {
    failures.push(`Canonical chart-history requirement missing: ${requirement}`);
  }
}
if (!app.includes('/api/v1/chart-history/{instrument_id}')) {
  failures.push("Version-checked chart-history API is missing.");
}
for (const variable of [
  "LIVE_CHART_HISTORY_PRIMARY_LIMIT",
  "LIVE_CHART_HISTORY_CONTEXT_LIMIT",
  "LIVE_CHART_HISTORY_HOURLY_LIMIT",
]) {
  if (!settings.includes(variable) || !env.includes(variable)) {
    failures.push(`Chart-history setting missing: ${variable}`);
  }
}
if (!env.includes("LIVE_HISTORY_LOOKBACK_DAYS=14")) {
  failures.push("The default canonical history bootstrap must cover 14 calendar days.");
}

const packageJson = JSON.parse(packageSource);
if (!packageJson.scripts["check:step12:time-history"]) {
  failures.push("Missing package script: check:step12:time-history");
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step12:time-history")) {
    failures.push(`${command} does not include the time/history regression gate.`);
  }
}

if (failures.length) {
  console.error("Step 12 time/history check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 12 time/history check passed (${requiredPaths.length} required paths).`);

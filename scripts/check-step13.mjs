import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "config/nse-calendar-through-2026-08-25.json",
  "services/backend/src/nifty_terminal/history/sources/angelone_source.py",
  "services/backend/src/nifty_terminal/history/session_normalizer.py",
  "services/backend/src/nifty_terminal/cli/acquire_angelone_history.py",
  "services/backend/tests/test_angelone_history_acquisition.py",
  "scripts/check-step13.mjs",
  "STEP-13-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 13 path: ${path}`);
  }
}

const [source, normalizer, cli, calendarLoader, calendarSource, provider, repository, packageSource] =
  await Promise.all([
    readFile("services/backend/src/nifty_terminal/history/sources/angelone_source.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/history/session_normalizer.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/cli/acquire_angelone_history.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/history/calendar_loader.py", "utf8"),
    readFile("config/nse-calendar-through-2026-08-25.json", "utf8"),
    readFile("services/backend/src/nifty_terminal/providers/angelone.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/history/sqlite_repository.py", "utf8"),
    readFile("package.json", "utf8"),
  ]);

for (const requirement of [
  "AngelOneHistoricalAcquirer",
  "fetch_finalized_minutes",
  "request_delay_milliseconds",
  "conflicting finalized rows",
  "NIFTY 50 spot volume must remain null",
  "source_sha256",
]) {
  if (!source.includes(requirement)) {
    failures.push(`Historical acquisition requirement missing: ${requirement}`);
  }
}
for (const requirement of [
  "normalize_to_continuous_sessions",
  "diagnose_expected_minute_coverage",
  "MINIMUM_EXPECTED_MINUTE_COVERAGE",
  "missing_bars_are_filled\": False",
]) {
  if (!normalizer.includes(requirement)) {
    failures.push(`Historical session-normalization requirement missing: ${requirement}`);
  }
}
for (const requirement of [
  "HistoricalImportPipeline",
  "HistoricalFeatureMaterializer",
  "TrainingDatasetAssembler",
  "training_research_ready",
  "approved_for_live_inference\": False",
  "LIVE_SIGNAL_KILL_SWITCH",
]) {
  if (!cli.includes(requirement)) {
    failures.push(`Step 13 CLI safety requirement missing: ${requirement}`);
  }
}
if (!calendarLoader.includes("validate_calendar_coverage")) {
  failures.push("Historical acquisition must enforce explicit calendar coverage.");
}
const calendar = JSON.parse(calendarSource);
if (
  calendar.exchange !== "NSE"
  || calendar.segment !== "CAPITAL_MARKET"
  || calendar.verified_through !== "2026-08-25"
  || !calendar.special_sessions?.["2025-02-01"]
  || !calendar.special_sessions?.["2025-10-21"]
  || !calendar.special_sessions?.["2026-02-01"]
  || !calendar.ignored_provider_observation_windows?.["2025-10-21"]
) {
  failures.push("The sourced NSE calendar metadata or special session is incomplete.");
}
if (!provider.includes("build_angelone_adapter")) {
  failures.push("Live and historical paths must share the same Angel One adapter factory.");
}
if (!repository.includes("timeframe: Timeframe | None")) {
  failures.push("Feature-row audit counts must be timeframe-specific.");
}

const packageJson = JSON.parse(packageSource);
for (const script of [
  "check:step13",
  "acquire:history:angelone",
  "acquire:history:angelone:windows",
]) {
  if (!packageJson.scripts[script]) failures.push(`Missing package script: ${script}`);
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step13")) {
    failures.push(`${command} does not include the Step 13 gate.`);
  }
}

if (failures.length) {
  console.error("Step 13 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 13 check passed (${requiredPaths.length} required paths).`);

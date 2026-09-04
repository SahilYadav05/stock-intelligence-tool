import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "config/nse-calendar-through-2026-08-25.json",
  "services/backend/src/nifty_terminal/calendar/nse.py",
  "services/backend/src/nifty_terminal/history/session_normalizer.py",
  "services/backend/src/nifty_terminal/runtime/live_market.py",
  "services/backend/src/nifty_terminal/cli/verify_angelone.py",
  "services/backend/tests/test_angelone_history_acquisition.py",
  "services/backend/tests/test_live_runtime.py",
  "services/backend/tests/test_ml_labels.py",
  "src/components/live-market-terminal.tsx",
  "STEP-13.3-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 13.3 path: ${path}`);
  }
}

const [calendarSource, calendarConfigSource, normalizer, runtime, verifier, cli, terminal, packageSource] =
  await Promise.all([
    readFile("services/backend/src/nifty_terminal/calendar/nse.py", "utf8"),
    readFile("config/nse-calendar-through-2026-08-25.json", "utf8"),
    readFile("services/backend/src/nifty_terminal/history/session_normalizer.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/runtime/live_market.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/cli/verify_angelone.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/cli/acquire_angelone_history.py", "utf8"),
    readFile("src/components/live-market-terminal.tsx", "utf8"),
    readFile("package.json", "utf8"),
  ]);

for (const requirement of [
  "class MarketPhase",
  "CLOSING_AUCTION_REFERENCE",
  "CLOSING_AUCTION_ORDER_ENTRY",
  "CLOSING_AUCTION_MATCHING",
  "date(2026, 8, 3)",
  "time(15, 15)",
  "time(15, 35)",
]) {
  if (!calendarSource.includes(requirement)) {
    failures.push(`CAS calendar requirement missing: ${requirement}`);
  }
}

const calendarConfig = JSON.parse(calendarConfigSource);
if (
  !calendarConfig.continuous_session_rules?.some(
    (rule) => rule.effective_from === "2026-08-03" && rule.close === "15:15",
  )
  || !calendarConfig.closing_auction_rules?.some(
    (rule) => rule.effective_from === "2026-08-03" && rule.matching_ends === "15:35",
  )
) {
  failures.push("The sourced calendar lacks the date-effective NSE CAS rules.");
}

if (!normalizer.includes("CLOSING_AUCTION_OBSERVATION_NOT_CONTINUOUS_CANDLE")) {
  failures.push("Historical CAS observations must be separated from continuous candles.");
}
for (const requirement of [
  "live_auction_observations",
  "historical_auction_observations",
  "NSE_CLOSING_AUCTION_ACTIVE_STANDARD_SIGNAL_DISABLED",
  "PREDICTION_HORIZON_CROSSES_CONTINUOUS_SESSION_CLOSE",
]) {
  if (!runtime.includes(requirement)) {
    failures.push(`Live CAS safety requirement missing: ${requirement}`);
  }
}
if (!verifier.includes("separate closing-auction observations")) {
  failures.push("Angel One verification must distinguish CAS observations.");
}
if (
  !cli.includes("auction_observations_are_model_inputs\": False")
  || !cli.includes("standard_60m_signal_last_decision\": \"14:15 IST\"")
) {
  failures.push("The historical report must record the CAS model-input policy.");
}
for (const requirement of [
  "CLOSING AUCTION · STANDARD SIGNAL DISABLED",
  "auction values excluded from standard signals",
]) {
  if (!terminal.includes(requirement)) {
    failures.push(`Dashboard CAS requirement missing: ${requirement}`);
  }
}

const packageJson = JSON.parse(packageSource);
if (!packageJson.scripts["check:step13:cas"]) {
  failures.push("Missing package script: check:step13:cas");
}
for (const command of ["test", "test:windows"]) {
  if (!packageJson.scripts[command]?.includes("check:step13:cas")) {
    failures.push(`${command} does not include the Step 13.3 CAS gate.`);
  }
}

if (failures.length) {
  console.error("Step 13.3 CAS check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 13.3 CAS check passed (${requiredPaths.length} required paths).`);

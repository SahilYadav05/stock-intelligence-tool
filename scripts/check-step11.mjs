import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "services/backend/src/nifty_terminal/providers/angelone.py",
  "services/backend/src/nifty_terminal/cli/verify_angelone.py",
  "services/backend/tests/test_angelone_provider.py",
  "scripts/check-step11.mjs",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 11 path: ${path}`);
  }
}

const [provider, settings, instruments, env, project, packageSource, gitignore] =
  await Promise.all([
    readFile("services/backend/src/nifty_terminal/providers/angelone.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/settings.py", "utf8"),
    readFile("services/backend/src/nifty_terminal/domain/instruments.py", "utf8"),
    readFile(".env.example", "utf8"),
    readFile("services/backend/pyproject.toml", "utf8"),
    readFile("package.json", "utf8"),
    readFile(".gitignore", "utf8"),
  ]);

for (const requirement of [
  "AngelOneProviderAdapter",
  "fetch_finalized_minutes",
  "provider_sequence_is_contiguous=False",
  "cumulative_volume=None",
  "close_connection",
]) {
  if (!provider.includes(requirement)) {
    failures.push(`Angel One adapter requirement missing: ${requirement}`);
  }
}
for (const variable of [
  "ANGELONE_API_KEY",
  "ANGELONE_CLIENT_CODE",
  "ANGELONE_PIN",
  "ANGELONE_TOTP_SECRET",
  "ANGELONE_NIFTY_WEBSOCKET_TOKEN",
  "ANGELONE_NIFTY_HISTORICAL_TOKEN",
]) {
  if (!settings.includes(variable) || !env.includes(variable)) {
    failures.push(`Angel One server setting missing: ${variable}`);
  }
}
if (!instruments.includes('provider="angelone"') || !instruments.includes("NSE:99926000")) {
  failures.push("Canonical NIFTY 50 Angel One mapping is missing.");
}
if (!project.includes('smartapi-python==1.5.5')) {
  failures.push("The verified Angel One SDK version is not pinned.");
}
if (!gitignore.includes("*.log")) {
  failures.push("Provider SDK logs must be excluded from source control.");
}

const packageJson = JSON.parse(packageSource);
for (const script of ["check:step11", "verify:angelone:windows"]) {
  if (!packageJson.scripts[script]) failures.push(`Missing package script: ${script}`);
}
if (!packageJson.scripts["backend:dev:windows"].includes("--env-file .env")) {
  failures.push("Windows backend startup must load the local .env explicitly.");
}

if (failures.length) {
  console.error("Step 11 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 11 check passed (${requiredPaths.length} required paths).`);

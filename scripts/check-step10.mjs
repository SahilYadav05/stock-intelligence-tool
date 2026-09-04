import { access, readFile } from "node:fs/promises";

const requiredPaths = [
  "contracts/release-readiness.v1.schema.json",
  "contracts/artifact-manifest.v1.schema.json",
  "contracts/security-audit-event.v1.schema.json",
  "contracts/drift-evidence.v1.schema.json",
  "services/backend/src/nifty_terminal/hardening/models.py",
  "services/backend/src/nifty_terminal/hardening/release.py",
  "services/backend/src/nifty_terminal/hardening/drift.py",
  "services/backend/src/nifty_terminal/hardening/security.py",
  "services/backend/src/nifty_terminal/hardening/circuit_breaker.py",
  "services/backend/src/nifty_terminal/hardening/audit.py",
  "services/backend/tests/test_hardening.py",
  "src/lib/hardening-contracts.ts",
  "src/lib/hardening-client.ts",
  "src/components/readiness-panel.tsx",
  "scripts/benchmark-backend.py",
  "SECURITY-DEPLOYMENT-RUNBOOK.md",
  "STEP-10-COMMAND-PROMPT.md",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 10 path: ${path}`);
  }
}

for (const schemaPath of requiredPaths.filter((path) => path.startsWith("contracts/"))) {
  try {
    const schema = JSON.parse(await readFile(schemaPath, "utf8"));
    if (!schema.$id?.endsWith(".v1")) failures.push(`Unversioned contract: ${schemaPath}`);
  } catch (error) {
    failures.push(`Invalid JSON contract ${schemaPath}: ${error.message}`);
  }
}

const [api, settings, release, security, monitoring, readiness, env, packageSource] = await Promise.all([
  readFile("services/backend/src/nifty_terminal/api/app.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/settings.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/hardening/release.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/hardening/security.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/tracking/monitoring.py", "utf8"),
  readFile("src/components/readiness-panel.tsx", "utf8"),
  readFile(".env.example", "utf8"),
  readFile("package.json", "utf8"),
]);

for (const route of ["/api/v1/live", "/api/v1/ready", "/api/v1/security/status"]) {
  if (!api.includes(route)) failures.push(`Hardened API route missing: ${route}`);
}
if (!security.includes("compare_digest") || !security.includes("RATE_LIMIT_EXCEEDED")) {
  failures.push("Constant-time bearer authentication and rate limiting are required.");
}
if (!settings.includes("Production requires API_AUTH_MODE=bearer") || !settings.includes("https://")) {
  failures.push("Production configuration must fail closed on authentication and HTTPS origins.");
}
for (const gate of [
  "CHART_MODEL_SNAPSHOT_NOT_SYNCHRONIZED",
  "APPROVED_RELEASE_MANIFEST_MISSING",
  "MODEL_ARTIFACT_MISSING_OR_HASH_MISMATCH",
  "DRIFT_REFERENCE_EVIDENCE_MISSING",
  "LIVE_SIGNAL_KILL_SWITCH_ACTIVE",
]) {
  if (!release.includes(gate)) failures.push(`Release gate missing: ${gate}`);
}
if (!monitoring.includes("No versioned reference distribution")) {
  failures.push("Monitoring must not infer drift readiness from outcome sample count.");
}
if (!readiness.includes("No gate can be overridden") || !readiness.includes("signal_allowed")) {
  failures.push("Dashboard readiness panel is not fail closed.");
}
if (!env.includes("API_AUTH_MODE") || !env.includes("LIVE_SIGNAL_KILL_SWITCH")) {
  failures.push("Environment template is missing hardening settings.");
}
const packageJson = JSON.parse(packageSource);
if (packageJson.version !== "1.0.0" || !packageJson.scripts["check:step10"]) {
  failures.push("Step 10 version or verification script is missing.");
}

if (failures.length) {
  console.error("Step 10 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 10 check passed (${requiredPaths.length} required paths).`);

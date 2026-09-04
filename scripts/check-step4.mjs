import { access, readFile } from "node:fs/promises";
import process from "node:process";

const requiredPaths = [
  "contracts/market-state-view.v1.schema.json",
  "contracts/websocket-message.v1.schema.json",
  "services/backend/src/nifty_terminal/api/app.py",
  "services/backend/src/nifty_terminal/api/messages.py",
  "services/backend/src/nifty_terminal/delivery/models.py",
  "services/backend/src/nifty_terminal/delivery/read_model.py",
  "services/backend/src/nifty_terminal/delivery/hub.py",
  "services/backend/src/nifty_terminal/delivery/service.py",
  "src/components/live-market-terminal.tsx",
  "src/lib/market-contracts.ts",
  "src/lib/market-stream.ts",
  "services/backend/tests/test_api.py",
  "services/backend/tests/test_delivery.py",
];

const failures = [];
for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    failures.push(`Missing Step 4 path: ${path}`);
  }
}

for (const schemaPath of [
  "contracts/market-state-view.v1.schema.json",
  "contracts/websocket-message.v1.schema.json",
]) {
  try {
    const schema = JSON.parse(await readFile(schemaPath, "utf8"));
    if (!schema.$id?.endsWith(".v1") || schema.properties?.schema_version?.const !== 1) {
      failures.push(`Contract is not explicitly versioned: ${schemaPath}`);
    }
  } catch (error) {
    failures.push(`Invalid JSON contract ${schemaPath}: ${error.message}`);
  }
}

const [terminal, api, delivery] = await Promise.all([
  readFile("src/components/live-market-terminal.tsx", "utf8"),
  readFile("services/backend/src/nifty_terminal/api/app.py", "utf8"),
  readFile("services/backend/src/nifty_terminal/delivery/models.py", "utf8"),
]);

for (const requiredText of [
  "LIVE ANALYSIS UNAVAILABLE",
  "SYNCING ANALYSIS",
  "No prices are being fabricated",
  "lightweight-charts",
]) {
  if (!terminal.includes(requiredText)) failures.push(`Terminal is missing: ${requiredText}`);
}
if (!api.includes("/ws/v1/market-state") || !api.includes("/api/v1/market-state")) {
  failures.push("Backend must expose the versioned HTTP and WebSocket market-state routes.");
}
if (!delivery.includes("model_input_candle_ids") || !delivery.includes("missing")) {
  failures.push("Delivery boundary must reject snapshot/candle mismatches.");
}
if (/\b25[0-9]{3}(?:\.\d+)?\b/.test(terminal)) {
  failures.push("Frontend contains a hard-coded NIFTY-like price.");
}

if (failures.length) {
  console.error("Step 4 check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Step 4 check passed (${requiredPaths.length} required paths).`);

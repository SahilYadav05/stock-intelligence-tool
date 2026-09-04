export type ReleaseStatus = "READY" | "BLOCKED";

export interface ReleaseReadinessContract {
  schema_version: 1;
  evaluated_at: string;
  status: ReleaseStatus;
  signal_allowed: boolean;
  blockers: string[];
  warnings: string[];
  release_id: string | null;
  exact_snapshot_match: boolean;
  security_mode: "DISABLED" | "BEARER";
  kill_switch_active: boolean;
}

export function parseReleaseReadiness(value: unknown): ReleaseReadinessContract | null {
  const candidate = isRecord(value) && isRecord(value.detail) ? value.detail : value;
  if (!isRecord(candidate) || candidate.schema_version !== 1) return null;
  if (!["READY", "BLOCKED"].includes(String(candidate.status))) return null;
  if (typeof candidate.signal_allowed !== "boolean") return null;
  if (!Array.isArray(candidate.blockers) || !candidate.blockers.every((item) => typeof item === "string")) return null;
  if (!Array.isArray(candidate.warnings) || !candidate.warnings.every((item) => typeof item === "string")) return null;
  if (typeof candidate.exact_snapshot_match !== "boolean") return null;
  if (!["DISABLED", "BEARER"].includes(String(candidate.security_mode))) return null;
  if (typeof candidate.kill_switch_active !== "boolean") return null;
  return candidate as unknown as ReleaseReadinessContract;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

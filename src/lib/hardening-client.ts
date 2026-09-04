import { parseReleaseReadiness, type ReleaseReadinessContract } from "@/src/lib/hardening-contracts";

export async function fetchReleaseReadiness(
  apiBaseUrl: string,
  signal: AbortSignal,
): Promise<ReleaseReadinessContract | null> {
  try {
    const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/api/v1/ready`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    });
    if (!response.ok && response.status !== 503) return null;
    return parseReleaseReadiness(await response.json());
  } catch {
    return null;
  }
}

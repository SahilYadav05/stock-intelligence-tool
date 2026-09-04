import {
  parseAnalysisAvailability,
  type AnalysisAvailabilityContract,
} from "@/src/lib/analysis-contracts";

export async function fetchSnapshotAnalysis(
  apiBaseUrl: string,
  snapshotId: string,
  revisionChecksum: string,
  signal?: AbortSignal,
): Promise<AnalysisAvailabilityContract | null> {
  const normalizedBase = apiBaseUrl.replace(/\/$/, "");
  const query = new URLSearchParams({ snapshot_id: snapshotId, timeframe: "5m" });
  try {
    const response = await fetch(
      `${normalizedBase}/api/v1/analysis/NIFTY50_SPOT?${query.toString()}`,
      { headers: { Accept: "application/json" }, signal },
    );
    if (!response.ok) return null;
    return parseAnalysisAvailability(
      await response.json(),
      snapshotId,
      revisionChecksum,
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    return null;
  }
}

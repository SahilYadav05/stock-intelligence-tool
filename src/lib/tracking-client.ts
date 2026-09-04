import { parseTrackingOverview, type TrackingOverviewContract } from "@/src/lib/tracking-contracts";

export async function fetchTrackingOverview(
  apiBaseUrl: string,
  signal: AbortSignal,
): Promise<TrackingOverviewContract | null> {
  try {
    const response = await fetch(
      `${apiBaseUrl.replace(/\/$/, "")}/api/v1/tracking/NIFTY50_SPOT?timeframe=5m`,
      { cache: "no-store", headers: { Accept: "application/json" }, signal },
    );
    if (!response.ok) return null;
    return parseTrackingOverview(await response.json());
  } catch {
    return null;
  }
}

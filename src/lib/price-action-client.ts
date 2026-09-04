import {
  parsePriceActionAvailability,
  type PriceActionAvailabilityContract,
} from "@/src/lib/price-action-contracts";

export async function fetchPriceAction(
  apiBaseUrl: string,
  snapshotId: string,
  revisionChecksum: string,
  signal?: AbortSignal,
): Promise<PriceActionAvailabilityContract | null> {
  const normalizedBase = apiBaseUrl.replace(/\/$/, "");
  const query = new URLSearchParams({ snapshot_id: snapshotId, timeframe: "5m" });
  try {
    const response = await fetch(
      `${normalizedBase}/api/v1/price-action/NIFTY50_SPOT?${query.toString()}`,
      { headers: { Accept: "application/json" }, signal },
    );
    if (!response.ok) return null;
    return parsePriceActionAvailability(
      await response.json(),
      snapshotId,
      revisionChecksum,
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    return null;
  }
}

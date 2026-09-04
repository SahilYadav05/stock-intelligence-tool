import type { CandleContract } from "@/src/lib/market-contracts";

export interface ChartHistoryContract {
  schema_version: 1;
  instrument_id: "NIFTY50_SPOT";
  exchange_timezone: "Asia/Kolkata";
  decision_time: string;
  data_as_of: string;
  candle_revision_checksum: string;
  candles: CandleContract[];
}

export async function fetchChartHistory(
  apiBaseUrl: string,
  decisionTime: string,
  revisionChecksum: string,
  signal?: AbortSignal,
): Promise<ChartHistoryContract | null> {
  const normalizedBase = apiBaseUrl.replace(/\/$/, "");
  const query = new URLSearchParams({
    timeframe: "5m",
    decision_time: decisionTime,
    candle_revision_checksum: revisionChecksum,
  });
  try {
    const response = await fetch(
      `${normalizedBase}/api/v1/chart-history/NIFTY50_SPOT?${query.toString()}`,
      { headers: { Accept: "application/json" }, signal },
    );
    if (!response.ok) return null;
    const value: unknown = await response.json();
    return isChartHistory(value, decisionTime, revisionChecksum) ? value : null;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    return null;
  }
}

export function mergeCanonicalCandles(
  historical: CandleContract[],
  current: CandleContract[],
): CandleContract[] {
  const byBucket = new Map<string, CandleContract>();
  for (const candle of [...historical, ...current]) {
    const key = `${candle.timeframe}:${candle.opens_at}`;
    const existing = byBucket.get(key);
    if (!existing || candle.revision >= existing.revision) byBucket.set(key, candle);
  }
  return [...byBucket.values()].sort(
    (left, right) => new Date(left.opens_at).getTime() - new Date(right.opens_at).getTime(),
  );
}

function isChartHistory(
  value: unknown,
  expectedDecisionTime: string,
  expectedChecksum: string,
): value is ChartHistoryContract {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (value.instrument_id !== "NIFTY50_SPOT") return false;
  if (value.exchange_timezone !== "Asia/Kolkata") return false;
  if (value.decision_time !== expectedDecisionTime) return false;
  if (value.candle_revision_checksum !== expectedChecksum) return false;
  if (!Array.isArray(value.candles)) return false;
  return value.candles.every(isFinalizedCandle);
}

function isFinalizedCandle(value: unknown): value is CandleContract {
  if (!isRecord(value)) return false;
  return (
    value.status === "FINALIZED"
    && typeof value.candle_id === "string"
    && typeof value.opens_at === "string"
    && typeof value.closes_at === "string"
    && typeof value.open === "string"
    && typeof value.high === "string"
    && typeof value.low === "string"
    && typeof value.close === "string"
    && ["5m", "15m", "1h"].includes(String(value.timeframe))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

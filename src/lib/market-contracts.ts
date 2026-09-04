export type DataStatus =
  | "DISCONNECTED"
  | "CONNECTING"
  | "RECOVERING"
  | "LIVE"
  | "DELAYED"
  | "STALE"
  | "MARKET_CLOSED";

export type Timeframe = "1m" | "5m" | "15m" | "1h";

export interface CandleContract {
  schema_version: 1;
  candle_id: string;
  instrument_id: string;
  timeframe: Timeframe;
  opens_at: string;
  closes_at: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string | null;
  status: "DEVELOPING" | "FINALIZED";
  revision: number;
  source: "PROVISIONAL_EVENTS" | "AUTHORITATIVE_MINUTE" | "AGGREGATED";
  provider: string;
  source_revision: number;
  finalized_at: string | null;
  component_candle_ids: string[];
  source_watermark: string;
  supersedes_candle_id: string | null;
}

export interface MarketStateSnapshotContract {
  schema_version: 1;
  snapshot_id: string;
  instrument_id: string;
  decision_time: string;
  created_at: string;
  data_as_of: string;
  data_mode: "REPLAY" | "LIVE";
  data_status: DataStatus;
  primary_timeframe: "5m";
  primary_candle_id: string;
  context_15m_candle_id: string | null;
  context_1h_candle_id: string | null;
  recent_primary_candle_ids: string[];
  developing_candle_id: string | null;
  model_input_candle_ids: string[];
  source_watermark: string;
  candle_revision_checksum: string;
  live_inference_eligible: boolean;
  blockers: string[];
}

export interface MarketStateViewContract {
  schema_version: 1;
  sync_state: "SYNCED";
  published_at: string;
  snapshot: MarketStateSnapshotContract;
  finalized_candles: CandleContract[];
  developing_candle: CandleContract | null;
}

export interface MarketStateMessage {
  schema_version: 1;
  message_type: "MARKET_STATE";
  sequence: number;
  sent_at: string;
  instrument_id: string;
  payload: MarketStateViewContract;
}

export interface StatusMessage {
  schema_version: 1;
  message_type: "STATUS";
  sequence: number;
  sent_at: string;
  instrument_id: string;
  payload: {
    data_status: DataStatus;
    reason: string;
    live_analysis_available: false;
  };
}

export type MarketSocketMessage = MarketStateMessage | StatusMessage;

export function parseMarketSocketMessage(value: unknown): MarketSocketMessage | null {
  if (!isRecord(value) || value.schema_version !== 1) return null;
  if (value.message_type === "STATUS") {
    if (!isRecord(value.payload) || !isDataStatus(value.payload.data_status)) return null;
    return value as unknown as StatusMessage;
  }
  if (value.message_type === "MARKET_STATE" && isMarketStateView(value.payload)) {
    return value as unknown as MarketStateMessage;
  }
  return null;
}

export function isMarketStateView(value: unknown): value is MarketStateViewContract {
  if (!isRecord(value) || value.schema_version !== 1 || value.sync_state !== "SYNCED") {
    return false;
  }
  if (!isRecord(value.snapshot) || !Array.isArray(value.finalized_candles)) return false;
  const snapshot = value.snapshot;
  if (typeof snapshot.snapshot_id !== "string" || snapshot.primary_timeframe !== "5m") {
    return false;
  }
  const candleIds = new Set(
    value.finalized_candles
      .filter(isRecord)
      .map((candle) => candle.candle_id)
      .filter((id): id is string => typeof id === "string"),
  );
  if (!candleIds.has(snapshot.primary_candle_id as string)) return false;
  if (!Array.isArray(snapshot.model_input_candle_ids)) return false;
  return snapshot.model_input_candle_ids.every(
    (candleId) => typeof candleId === "string" && candleIds.has(candleId),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isDataStatus(value: unknown): value is DataStatus {
  return [
    "DISCONNECTED",
    "CONNECTING",
    "RECOVERING",
    "LIVE",
    "DELAYED",
    "STALE",
    "MARKET_CLOSED",
  ].includes(String(value));
}

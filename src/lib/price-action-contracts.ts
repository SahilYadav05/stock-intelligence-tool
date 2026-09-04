export type PriceActionBias = "BULLISH" | "BEARISH" | "NEUTRAL" | "UNAVAILABLE";
export type PriceActionSetup =
  | "BUY_TRIGGER"
  | "SELL_TRIGGER"
  | "BULLISH_WATCH"
  | "BEARISH_WATCH"
  | "WAIT"
  | "UNAVAILABLE";

export interface PriceActionLevelContract {
  price: string;
  kind: string;
  strength: number;
  touches: number;
}

export interface ConditionalTradePlanContract {
  direction: "BUY" | "SELL";
  trigger: string;
  entry_low: string;
  entry_high: string;
  stop: string;
  invalidation: string;
  target1: string;
  target2: string;
  target3: string;
  risk_points: string;
  target1_reward_risk: number;
  target2_reward_risk: number;
  target3_reward_risk: number;
  expiry_bars: number;
  blockers: string[];
}

export interface PriceActionAnalysisContract {
  schema_version: 1;
  snapshot_id: string;
  candle_revision_checksum: string;
  instrument_id: "NIFTY50_SPOT";
  decision_time: string;
  generated_at: string;
  version: string;
  bias: PriceActionBias;
  setup: PriceActionSetup;
  confluence_score: number;
  evidence_grade: "STRONG" | "MODERATE" | "WEAK" | "NO_EDGE" | "UNAVAILABLE";
  structure_5m: string;
  trend_5m: string;
  trend_15m: string;
  trend_1h: string;
  volatility_regime: string;
  patterns: string[];
  support_levels: PriceActionLevelContract[];
  resistance_levels: PriceActionLevelContract[];
  reasons: string[];
  contradictory_evidence: string[];
  trade_plan: ConditionalTradePlanContract | null;
  blockers: string[];
  research_only: true;
  official_signal: false;
  calibrated_probability: null;
  automatic_execution: false;
}

export interface PriceActionAvailabilityContract {
  schema_version: 1;
  sync_state: "SYNCED" | "SYNCING";
  snapshot_id: string;
  candle_revision_checksum: string;
  analysis: PriceActionAnalysisContract | null;
  reason: string | null;
}

export function parsePriceActionAvailability(
  value: unknown,
  expectedSnapshotId: string,
  expectedRevisionChecksum: string,
): PriceActionAvailabilityContract | null {
  if (!isRecord(value) || value.schema_version !== 1) return null;
  if (!['SYNCED', 'SYNCING'].includes(String(value.sync_state))) return null;
  if (value.snapshot_id !== expectedSnapshotId) return null;
  if (value.candle_revision_checksum !== expectedRevisionChecksum) return null;
  if (value.sync_state === "SYNCING") {
    return value.analysis === null ? value as unknown as PriceActionAvailabilityContract : null;
  }
  if (!isAnalysis(value.analysis, expectedSnapshotId, expectedRevisionChecksum)) return null;
  return value as unknown as PriceActionAvailabilityContract;
}

function isAnalysis(value: unknown, snapshotId: string, checksum: string) {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (value.snapshot_id !== snapshotId || value.candle_revision_checksum !== checksum) return false;
  if (!['BULLISH', 'BEARISH', 'NEUTRAL', 'UNAVAILABLE'].includes(String(value.bias))) return false;
  if (!Array.isArray(value.support_levels) || !Array.isArray(value.resistance_levels)) return false;
  if (!Array.isArray(value.reasons) || !Array.isArray(value.contradictory_evidence)) return false;
  if (value.research_only !== true || value.official_signal !== false) return false;
  return value.calibrated_probability === null && value.automatic_execution === false;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

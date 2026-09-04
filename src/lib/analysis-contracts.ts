export type SignalDirection = "BUY" | "SELL" | "WAIT";

export interface RiskLevelsContract {
  entry_low: string;
  entry_high: string;
  stop: string;
  invalidation: string;
  target1: string;
  target2: string;
  target3: string;
  target1_reward_risk: number;
}

export interface SignalDecisionContract {
  schema_version: 1;
  signal_id: string;
  prediction_id: string;
  calibration_id: string;
  snapshot_id: string;
  instrument_id: "NIFTY50_SPOT";
  decision_time: string;
  created_at: string;
  expires_at: string;
  direction: SignalDirection;
  lifecycle_status:
    | "NO_SIGNAL"
    | "SETUP_DETECTED"
    | "WAITING_FOR_CONFIRMATION"
    | "ACTIVE"
    | "TARGET_HIT"
    | "STOP_HIT"
    | "INVALIDATED"
    | "EXPIRED";
  probabilities: { DOWN: number; NEITHER: number; UP: number } | null;
  expected_atr: number | null;
  risk_levels: RiskLevelsContract | null;
  blockers: string[];
  signal_policy_version: string;
  risk_policy_version: string;
  input_revision_checksum: string;
  automatic_execution: false;
}

export interface HistoricalSignalContract {
  signal_id: string;
  occurred_at: string;
  direction: "BUY" | "SELL";
  price: string;
  status: string;
}

export interface AnalysisViewContract {
  schema_version: 1;
  analysis_id: string;
  snapshot_id: string;
  candle_revision_checksum: string;
  instrument_id: "NIFTY50_SPOT";
  decision_time: string;
  generated_at: string;
  data_as_of: string;
  signal: SignalDecisionContract;
  model_version: string;
  calibration_version: string;
  feature_version: string;
  market_context: {
    status: "AVAILABLE" | "UNAVAILABLE";
    regime: string | null;
    trend: string | null;
    momentum: string | null;
    volatility: string | null;
    support_levels: string[];
    resistance_levels: string[];
  };
  reasons: string[];
  contradictory_evidence: string[];
  news: {
    status: "AVAILABLE" | "NO_MATERIAL_EVENT" | "UNAVAILABLE";
    items: Array<{
      event_id: string;
      headline: string;
      source: string;
      published_at: string;
      received_at: string;
      impact: string;
    }>;
  };
  historical_analogs: { count: number | null; summary: string | null };
  historical_signals: HistoricalSignalContract[];
}

export interface AnalysisAvailabilityContract {
  schema_version: 1;
  sync_state: "SYNCED" | "SYNCING_ANALYSIS";
  snapshot_id: string;
  candle_revision_checksum: string;
  analysis: AnalysisViewContract | null;
  signal_suppressed: boolean;
  reason: string | null;
}

export function parseAnalysisAvailability(
  value: unknown,
  expectedSnapshotId: string,
  expectedRevisionChecksum: string,
): AnalysisAvailabilityContract | null {
  if (!isRecord(value) || value.schema_version !== 1) return null;
  if (!['SYNCED', 'SYNCING_ANALYSIS'].includes(String(value.sync_state))) return null;
  if (typeof value.snapshot_id !== "string") return null;
  if (typeof value.candle_revision_checksum !== "string") return null;
  if (value.snapshot_id !== expectedSnapshotId) return null;
  if (value.candle_revision_checksum !== expectedRevisionChecksum) return null;
  if (value.sync_state === "SYNCING_ANALYSIS") {
    return value.analysis === null ? (value as unknown as AnalysisAvailabilityContract) : null;
  }
  if (!isAnalysisView(value.analysis, expectedSnapshotId, expectedRevisionChecksum)) return null;
  return value as unknown as AnalysisAvailabilityContract;
}

function isAnalysisView(
  value: unknown,
  expectedSnapshotId: string,
  expectedRevisionChecksum: string,
): value is AnalysisViewContract {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (value.snapshot_id !== expectedSnapshotId) return false;
  if (value.candle_revision_checksum !== expectedRevisionChecksum) return false;
  if (!isRecord(value.signal) || value.signal.snapshot_id !== expectedSnapshotId) return false;
  if (value.signal.input_revision_checksum !== expectedRevisionChecksum) return false;
  if (!['BUY', 'SELL', 'WAIT'].includes(String(value.signal.direction))) return false;
  if (!Array.isArray(value.reasons) || !Array.isArray(value.contradictory_evidence)) return false;
  return true;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

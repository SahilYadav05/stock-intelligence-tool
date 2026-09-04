export type EvidenceStatus = "READY" | "INSUFFICIENT_SAMPLE" | "UNAVAILABLE" | "BREACHED";
export type MonitorStatus = "OK" | "WARN" | "CRITICAL" | "UNAVAILABLE";

export interface PredictionAnalyticsContract {
  schema_version: 1;
  instrument_id: "NIFTY50_SPOT";
  generated_at: string;
  minimum_sample: number;
  tracked_predictions: number;
  assessed_predictions: number;
  pending_predictions: number;
  actual_outcome_counts: { DOWN: number; NEITHER: number; UP: number };
  metrics_status: EvidenceStatus;
  accuracy: number | null;
  multiclass_brier_score: number | null;
  expected_calibration_error: number | null;
  paper_trades: number;
  closed_paper_trades: number;
  paper_metrics_status: EvidenceStatus;
  paper_win_rate: number | null;
  paper_total_points: string | null;
  blockers: string[];
  performance_claim_allowed: false;
}

export interface MonitoringCheckContract {
  key: string;
  status: MonitorStatus;
  observed_at: string;
  detail: string;
}

export interface MonitoringViewContract {
  schema_version: 1;
  instrument_id: "NIFTY50_SPOT";
  generated_at: string;
  overall_status: MonitorStatus;
  checks: MonitoringCheckContract[];
  model_drift_status: EvidenceStatus;
  probability_drift_status: EvidenceStatus;
  alerting_enabled: boolean;
}

export interface PaperTradeContract {
  schema_version: 1;
  paper_trade_id: string;
  signal_id: string;
  prediction_id: string;
  snapshot_id: string;
  instrument_id: "NIFTY50_SPOT";
  created_at: string;
  expires_at: string;
  direction: "BUY" | "SELL";
  status: "PLANNED";
  entry_low: string;
  entry_high: string;
  stop: string;
  target1: string;
  target2: string;
  target3: string;
  model_version: string;
  calibration_version: string;
  signal_policy_version: string;
  input_revision_checksum: string;
  unit: "NIFTY_INDEX_POINTS";
  automatic_execution: false;
}

export interface PaperTradeEventContract {
  schema_version: 1;
  event_id: string;
  paper_trade_id: string;
  signal_id: string;
  event_type: "CREATED" | "OPENED" | "TARGET_1_HIT" | "STOP_HIT" | "EXPIRED" | "INVALIDATED";
  status: "PLANNED" | "OPEN" | "TARGET_1_HIT" | "STOP_HIT" | "EXPIRED" | "INVALIDATED";
  occurred_at: string;
  observed_price: string | null;
  pnl_points: string | null;
  reason: string;
  unit: "NIFTY_INDEX_POINTS";
}

export interface TrackingOverviewContract {
  schema_version: 1;
  instrument_id: "NIFTY50_SPOT";
  generated_at: string;
  analytics: PredictionAnalyticsContract;
  monitoring: MonitoringViewContract;
  paper_trades: PaperTradeContract[];
  recent_paper_events: PaperTradeEventContract[];
  paper_only: true;
  automatic_execution: false;
}

export function parseTrackingOverview(value: unknown): TrackingOverviewContract | null {
  if (!isRecord(value) || value.schema_version !== 1) return null;
  if (value.instrument_id !== "NIFTY50_SPOT") return null;
  if (value.paper_only !== true || value.automatic_execution !== false) return null;
  if (!isAnalytics(value.analytics) || !isMonitoring(value.monitoring)) return null;
  if (!Array.isArray(value.paper_trades) || !Array.isArray(value.recent_paper_events)) return null;
  if (!value.paper_trades.every(isPaperTrade) || !value.recent_paper_events.every(isPaperEvent)) return null;
  return value as unknown as TrackingOverviewContract;
}

function isAnalytics(value: unknown): boolean {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (!evidenceStatuses.has(String(value.metrics_status))) return false;
  if (!evidenceStatuses.has(String(value.paper_metrics_status))) return false;
  for (const key of ["tracked_predictions", "assessed_predictions", "pending_predictions", "minimum_sample"]) {
    if (!Number.isInteger(value[key]) || Number(value[key]) < 0) return false;
  }
  if (
    value.metrics_status !== "READY"
    && (value.accuracy !== null
      || value.multiclass_brier_score !== null
      || value.expected_calibration_error !== null)
  ) return false;
  if (
    value.paper_metrics_status !== "READY"
    && (value.paper_win_rate !== null || value.paper_total_points !== null)
  ) return false;
  if (value.performance_claim_allowed !== false) return false;
  return isRecord(value.actual_outcome_counts) && Array.isArray(value.blockers);
}

function isMonitoring(value: unknown): boolean {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (value.instrument_id !== "NIFTY50_SPOT") return false;
  if (!monitorStatuses.has(String(value.overall_status))) return false;
  if (!Array.isArray(value.checks)) return false;
  return value.checks.every(
    (item) => isRecord(item) && typeof item.key === "string" && monitorStatuses.has(String(item.status)),
  );
}

function isPaperTrade(value: unknown): boolean {
  return isRecord(value)
    && value.schema_version === 1
    && value.instrument_id === "NIFTY50_SPOT"
    && ["BUY", "SELL"].includes(String(value.direction))
    && value.unit === "NIFTY_INDEX_POINTS"
    && value.automatic_execution === false;
}

function isPaperEvent(value: unknown): boolean {
  return isRecord(value)
    && value.schema_version === 1
    && value.unit === "NIFTY_INDEX_POINTS"
    && typeof value.event_id === "string";
}

const evidenceStatuses = new Set(["READY", "INSUFFICIENT_SAMPLE", "UNAVAILABLE", "BREACHED"]);
const monitorStatuses = new Set(["OK", "WARN", "CRITICAL", "UNAVAILABLE"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

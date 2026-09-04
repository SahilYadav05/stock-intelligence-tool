"use client";

import { useEffect, useMemo, useState } from "react";

import { ReadinessPanel } from "@/src/components/readiness-panel";
import { fetchTrackingOverview } from "@/src/lib/tracking-client";
import type {
  EvidenceStatus,
  PaperTradeContract,
  PaperTradeEventContract,
  TrackingOverviewContract,
} from "@/src/lib/tracking-contracts";

interface TrackingConsoleProps {
  apiBaseUrl: string;
}

export function TrackingConsole({ apiBaseUrl }: TrackingConsoleProps) {
  const [overview, setOverview] = useState<TrackingOverviewContract | null>(null);
  const [reason, setReason] = useState(
    apiBaseUrl ? "CONNECTING_TO_TRACKING_API" : "MARKET_API_NOT_CONFIGURED",
  );

  useEffect(() => {
    if (!apiBaseUrl) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const refresh = async () => {
      if (document.visibilityState === "hidden") {
        timer = setTimeout(refresh, 30_000);
        return;
      }
      const next = await fetchTrackingOverview(apiBaseUrl, controller.signal);
      if (stopped) return;
      if (next) {
        setOverview(next);
        setReason("TRACKING_OVERVIEW_READY");
      } else {
        setOverview(null);
        setReason("TRACKING_API_UNAVAILABLE");
      }
      timer = setTimeout(refresh, 10_000);
    };
    void refresh();
    return () => {
      stopped = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [apiBaseUrl]);

  const latestEvents = useMemo(() => {
    const byTrade = new Map<string, PaperTradeEventContract>();
    for (const event of overview?.recent_paper_events ?? []) {
      if (!byTrade.has(event.paper_trade_id)) byTrade.set(event.paper_trade_id, event);
    }
    return byTrade;
  }, [overview]);

  return (
    <section className="tracking-console" aria-label="Prediction tracking and paper research">
      <header className="tracking-console-header">
        <div>
          <p className="eyebrow">Steps 9–10 · Evidence and control plane</p>
          <h2>Prediction analytics, paper journal, monitoring & release safety</h2>
        </div>
        <div className="paper-disclaimer">
          <strong>PAPER ONLY</strong>
          <span>NIFTY index points · no orders · no cash-performance claim</span>
        </div>
      </header>

      <div className="observability-grid">
        <ResearchAnalytics overview={overview} reason={reason} />
        <PaperJournal overview={overview} latestEvents={latestEvents} reason={reason} />
        <MonitoringPanel overview={overview} reason={reason} />
        <ReadinessPanel apiBaseUrl={apiBaseUrl} />
      </div>
    </section>
  );
}

function ResearchAnalytics({ overview, reason }: PanelProps) {
  const analytics = overview?.analytics ?? null;
  const totalOutcomes = analytics
    ? Object.values(analytics.actual_outcome_counts).reduce((sum, value) => sum + value, 0)
    : 0;
  return (
    <article className="observability-panel" id="research">
      <PanelHeader
        eyebrow="Out-of-sample follow-through"
        title="Prediction analytics"
        status={analytics?.metrics_status ?? "UNAVAILABLE"}
      />
      <div className="metric-card-grid">
        <Metric label="Tracked" value={analytics ? String(analytics.tracked_predictions) : "—"} />
        <Metric label="Assessed" value={analytics ? String(analytics.assessed_predictions) : "—"} />
        <Metric label="Accuracy" value={formatPercent(analytics?.accuracy)} />
        <Metric label="Brier" value={formatMetric(analytics?.multiclass_brier_score)} />
        <Metric label="Calibration error" value={formatPercent(analytics?.expected_calibration_error)} />
        <Metric label="Pending" value={analytics ? String(analytics.pending_predictions) : "—"} />
      </div>
      <div className="outcome-distribution" aria-label="Assessed outcome distribution">
        {(["UP", "DOWN", "NEITHER"] as const).map((outcome) => {
          const value = analytics?.actual_outcome_counts[outcome] ?? 0;
          const width = totalOutcomes ? (value / totalOutcomes) * 100 : 0;
          return <div key={outcome}><span>{outcome}</span><i><b style={{ width: `${width}%` }} /></i><strong>{analytics ? value : "—"}</strong></div>;
        })}
      </div>
      <EvidenceFooter
        status={analytics?.metrics_status ?? "UNAVAILABLE"}
        count={analytics?.assessed_predictions ?? 0}
        minimum={analytics?.minimum_sample ?? 30}
        reason={reason}
      />
    </article>
  );
}

function PaperJournal({ overview, latestEvents, reason }: PanelProps & { latestEvents: Map<string, PaperTradeEventContract> }) {
  const trades = overview?.paper_trades ?? [];
  return (
    <article className="observability-panel paper-journal" id="journal">
      <PanelHeader
        eyebrow="Immutable simulated lifecycle"
        title="Paper journal"
        status={overview?.analytics.paper_metrics_status ?? "UNAVAILABLE"}
      />
      <div className="journal-summary">
        <Metric label="Plans" value={overview ? String(overview.analytics.paper_trades) : "—"} />
        <Metric label="Closed" value={overview ? String(overview.analytics.closed_paper_trades) : "—"} />
        <Metric label="Win rate" value={formatPercent(overview?.analytics.paper_win_rate)} />
        <Metric label="Total points" value={overview?.analytics.paper_total_points ?? "—"} />
      </div>
      {trades.length ? (
        <div className="journal-table-wrap">
          <table className="journal-table">
            <thead><tr><th>Time</th><th>Side</th><th>Entry</th><th>Status</th><th>Points</th></tr></thead>
            <tbody>
              {trades.slice(0, 8).map((trade) => <PaperTradeRow key={trade.paper_trade_id} trade={trade} event={latestEvents.get(trade.paper_trade_id)} />)}
            </tbody>
          </table>
        </div>
      ) : (
        <PanelEmpty title="No paper trades recorded" text={`${humanize(reason)}. WAIT never creates a paper position.`} />
      )}
      <p className="journal-note">Results use conservative fills and finalized candles. Ambiguous intrabar ordering is invalidated, never guessed.</p>
    </article>
  );
}

function MonitoringPanel({ overview, reason }: PanelProps) {
  const monitoring = overview?.monitoring ?? null;
  return (
    <article className="observability-panel" id="monitoring">
      <PanelHeader eyebrow="Fail-safe observability" title="System monitoring" status={monitoring?.overall_status ?? "UNAVAILABLE"} />
      {monitoring ? (
        <ul className="monitor-list">
          {monitoring.checks.map((check) => (
            <li key={check.key}><i className={`monitor-dot monitor-${check.status.toLowerCase()}`} /><div><strong>{humanize(check.key)}</strong><span>{check.detail}</span></div><em>{check.status}</em></li>
          ))}
        </ul>
      ) : <PanelEmpty title="Monitoring API unavailable" text={humanize(reason)} />}
      <div className="drift-grid">
        <div><span>Feature/model drift</span><strong>{monitoring?.model_drift_status ?? "UNAVAILABLE"}</strong></div>
        <div><span>Probability drift</span><strong>{monitoring?.probability_drift_status ?? "UNAVAILABLE"}</strong></div>
        <div><span>External alerts</span><strong>{monitoring?.alerting_enabled ? "ENABLED" : "DISABLED"}</strong></div>
      </div>
    </article>
  );
}

interface PanelProps { overview: TrackingOverviewContract | null; reason: string }

function PanelHeader({ eyebrow, title, status }: { eyebrow: string; title: string; status: EvidenceStatus | string }) {
  return <header className="observability-header"><div><p className="eyebrow">{eyebrow}</p><h3>{title}</h3></div><span className={`evidence-chip evidence-${status.toLowerCase()}`}>{status.replaceAll("_", " ")}</span></header>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="tracking-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function PaperTradeRow({ trade, event }: { trade: PaperTradeContract; event?: PaperTradeEventContract }) {
  return <tr><td>{formatIst(trade.created_at)}</td><td className={`side-${trade.direction.toLowerCase()}`}>{trade.direction}</td><td>{trade.entry_low}–{trade.entry_high}</td><td>{event?.status ?? trade.status}</td><td>{event?.pnl_points ?? "—"}</td></tr>;
}

function EvidenceFooter({ status, count, minimum, reason }: { status: EvidenceStatus; count: number; minimum: number; reason: string }) {
  const text = status === "READY" ? `${count} assessments passed the reporting gate.` : status === "INSUFFICIENT_SAMPLE" ? `${count}/${minimum} assessments · precise metrics suppressed.` : `${humanize(reason)} · no performance metric displayed.`;
  return <p className="evidence-footer">{text}</p>;
}

function PanelEmpty({ title, text }: { title: string; text: string }) {
  return <div className="tracking-empty"><i aria-hidden="true">·</i><div><strong>{title}</strong><p>{text}</p></div></div>;
}

function formatPercent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatMetric(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(4);
}

function formatIst(value: string) {
  return new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(value));
}

function humanize(value: string) {
  return value.toLowerCase().replaceAll("_", " ");
}

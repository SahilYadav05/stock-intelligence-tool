"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { TerminalChart, type IndicatorVisibility } from "@/src/components/terminal-chart";
import { TrackingConsole } from "@/src/components/tracking-console";
import { PROJECT, STAGE_STEPS } from "@/src/config/project";
import { fetchSnapshotAnalysis } from "@/src/lib/analysis-client";
import type {
  AnalysisAvailabilityContract,
  AnalysisViewContract,
  SignalDecisionContract,
} from "@/src/lib/analysis-contracts";
import type {
  CandleContract,
  DataStatus,
  MarketStateViewContract,
  Timeframe,
} from "@/src/lib/market-contracts";
import { connectMarketStream } from "@/src/lib/market-stream";
import { fetchChartHistory, mergeCanonicalCandles } from "@/src/lib/chart-history";
import { fetchPriceAction } from "@/src/lib/price-action-client";
import type {
  ConditionalTradePlanContract,
  PriceActionAnalysisContract,
  PriceActionAvailabilityContract,
} from "@/src/lib/price-action-contracts";

// Local development can point directly at a loopback API. In production this
// remains empty, then the browser uses the same-origin Worker gateway after
// hydration so no private origin or credential enters the client bundle.
const CONFIGURED_API_BASE_URL = process.env.NEXT_PUBLIC_MARKET_API_URL?.trim() ?? "";
const TIMEFRAMES: Array<{ label: string; value: Timeframe; role: string }> = [
  { label: "5m", value: "5m", role: "Signal" },
  { label: "15m", value: "15m", role: "Context" },
  { label: "1H", value: "1h", role: "Regime" },
];

interface LiveMarketTerminalProps {
  authenticated: boolean;
}

export function LiveMarketTerminal({ authenticated }: LiveMarketTerminalProps) {
  const apiBaseUrl = useMarketApiBaseUrl();
  const [view, setView] = useState<MarketStateViewContract | null>(null);
  const [analysisAvailability, setAnalysisAvailability] =
    useState<AnalysisAvailabilityContract | null>(null);
  const [priceActionAvailability, setPriceActionAvailability] =
    useState<PriceActionAvailabilityContract | null>(null);
  const [dataStatus, setDataStatus] = useState<DataStatus>("DISCONNECTED");
  const [marketReason, setMarketReason] = useState(
    CONFIGURED_API_BASE_URL ? "CONNECTING_TO_MARKET_API" : "MARKET_API_PROXY_CONFIGURING",
  );
  const [analysisReason, setAnalysisReason] = useState("NO_MARKET_SNAPSHOT");
  const [timeframe, setTimeframe] = useState<Timeframe>("5m");
  const [indicators, setIndicators] = useState<IndicatorVisibility>({
    ema20: true,
    ema50: false,
    levels: true,
    signals: true,
  });
  const [clockIso, setClockIso] = useState<string | null>(null);
  const [chartHistory, setChartHistory] = useState<CandleContract[]>([]);

  useEffect(() => {
    const updateClock = () => setClockIso(new Date().toISOString());
    updateClock();
    const timer = setInterval(updateClock, 1_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!apiBaseUrl) return;
    return connectMarketStream(apiBaseUrl, {
      onConnecting: () => {
        setDataStatus("CONNECTING");
        setMarketReason("CONNECTING_TO_MARKET_API");
      },
      onStatus: (status, nextReason) => {
        setDataStatus(status);
        setMarketReason(nextReason);
        if (status !== "LIVE") {
          setAnalysisAvailability(null);
          setAnalysisReason("MARKET_DATA_NOT_LIVE");
        }
      },
      onView: (nextView) => {
        setView(nextView);
        setDataStatus(nextView.snapshot.data_status);
        setMarketReason(
          nextView.snapshot.blockers.find((blocker) => blocker.includes("CLOSING_AUCTION"))
            ?? nextView.snapshot.blockers[0]
            ?? "CANONICAL_SNAPSHOT_RECEIVED",
        );
        setAnalysisAvailability(null);
        setPriceActionAvailability(null);
        setAnalysisReason(
          nextView.snapshot.live_inference_eligible
            ? "FETCHING_EXACT_SNAPSHOT_ANALYSIS"
            : nextView.snapshot.blockers[0] ?? "LIVE_INFERENCE_NOT_ELIGIBLE",
        );
      },
    });
  }, [apiBaseUrl]);

  useEffect(() => {
    if (
      !apiBaseUrl
      || !view
      || view.snapshot.data_status !== "LIVE"
      || !view.snapshot.live_inference_eligible
    ) return;
    const controller = new AbortController();
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const requestAnalysis = async () => {
      const availability = await fetchSnapshotAnalysis(
        apiBaseUrl,
        view.snapshot.snapshot_id,
        view.snapshot.candle_revision_checksum,
        controller.signal,
      );
      if (stopped) return;
      if (!availability) {
        setAnalysisAvailability(null);
        setAnalysisReason("ANALYSIS_API_UNAVAILABLE_OR_MISMATCHED");
        retryTimer = setTimeout(requestAnalysis, 2_000);
        return;
      }
      setAnalysisAvailability(availability);
      setAnalysisReason(availability.reason ?? "EXACT_SNAPSHOT_ANALYSIS_READY");
      if (availability.sync_state === "SYNCING_ANALYSIS") {
        retryTimer = setTimeout(requestAnalysis, 1_500);
      }
    };
    void requestAnalysis();
    return () => {
      stopped = true;
      controller.abort();
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [apiBaseUrl, view]);

  useEffect(() => {
    if (!apiBaseUrl || !view) return;
    const controller = new AbortController();
    const loadPriceAction = async () => {
      const availability = await fetchPriceAction(
        apiBaseUrl,
        view.snapshot.snapshot_id,
        view.snapshot.candle_revision_checksum,
        controller.signal,
      );
      if (availability) setPriceActionAvailability(availability);
    };
    void loadPriceAction();
    return () => controller.abort();
  }, [apiBaseUrl, view]);

  const decisionTime = view?.snapshot.decision_time ?? null;
  const candleRevisionChecksum = view?.snapshot.candle_revision_checksum ?? null;

  useEffect(() => {
    if (!apiBaseUrl || !decisionTime || !candleRevisionChecksum) return;
    const controller = new AbortController();
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const load = async () => {
      const history = await fetchChartHistory(
        apiBaseUrl,
        decisionTime,
        candleRevisionChecksum,
        controller.signal,
      );
      if (stopped) return;
      if (history) {
        setChartHistory(history.candles);
        return;
      }
      retryTimer = setTimeout(load, 2_000);
    };
    void load();
    return () => {
      stopped = true;
      controller.abort();
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [apiBaseUrl, decisionTime, candleRevisionChecksum]);

  const canonicalCandles = useMemo(
    () => mergeCanonicalCandles(chartHistory, view?.finalized_candles ?? []),
    [chartHistory, view],
  );

  const displayedCandles = useMemo(
    () => canonicalCandles.filter((candle) => candle.timeframe === timeframe),
    [canonicalCandles, timeframe],
  );
  const latest = displayedCandles.at(-1) ?? null;
  const previous = displayedCandles.at(-2) ?? null;
  const developing = timeframe === "5m" ? view?.developing_candle ?? null : null;
  const displayedQuote = developing ?? latest;
  const priceMove = priceChange(displayedQuote, developing ? latest : previous);
  const analysis = synchronizedAnalysis(view, analysisAvailability, dataStatus);
  const priceAction = synchronizedPriceAction(view, priceActionAvailability);
  const dataAge = view ? formatAge(view.snapshot.data_as_of) : "—";
  const closingAuctionActive = marketReason.includes("CLOSING_AUCTION");
  const marketPhaseLabel = closingAuctionActive
    ? "CLOSING AUCTION"
    : dataStatus.replace("_", " ");
  const snapshotShort = view ? view.snapshot.snapshot_id.slice(0, 12) : "—";
  const revisionShort = view
    ? view.snapshot.candle_revision_checksum.slice(0, 12)
    : "—";

  return (
    <main className="terminal-app" id="terminal">
      <header className="terminal-topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">NI</span>
          <div>
            <p className="eyebrow">Private trading intelligence</p>
            <h1>NIFTY Intelligence Terminal</h1>
          </div>
        </div>
        <nav className="terminal-nav" aria-label="Terminal sections">
          <a className="active" href="#terminal">Terminal</a>
          <a href="#research">Research</a>
          <a href="#journal">Journal</a>
          <a href="#monitoring">Monitoring</a>
          <a href="#readiness">Readiness</a>
        </nav>
        <div className="topbar-statuses">
          <span className="mode-chip">Research · No execution</span>
          <span className="identity-chip">
            <span className="identity-dot" aria-hidden="true" />
            {authenticated ? "Private session" : "Local session"}
          </span>
        </div>
      </header>

      <section className="market-toolbar" aria-label="Market controls">
        <button className="instrument-selector" type="button" aria-label="Selected instrument NIFTY 50">
          <span className="instrument-avatar">N50</span>
          <span>
            <small>Instrument</small>
            <strong>NIFTY 50</strong>
            <em>NSE · Spot index</em>
          </span>
          <i aria-hidden="true">⌄</i>
        </button>
        <div className="timeframe-control" aria-label="Chart timeframe">
          {TIMEFRAMES.map((item) => (
            <button
              className={timeframe === item.value ? "active" : ""}
              key={item.value}
              onClick={() => setTimeframe(item.value)}
              title={`${item.role} timeframe`}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="quote-strip">
          <div>
            <small>{developing ? "Current price" : "Last finalized"}</small>
            <strong>{displayedQuote ? formatPrice(displayedQuote.close) : "—"}</strong>
          </div>
          <div className={priceMove.direction}>
            <small>Move</small>
            <strong>{priceMove.label}</strong>
          </div>
          <div>
            <small>Candle closed</small>
            <strong>{latest ? formatIst(latest.closes_at) : "—"}</strong>
          </div>
        </div>
        <DataStatusBadge status={dataStatus} age={dataAge} label={marketPhaseLabel} />
      </section>

      <div className={`safety-ribbon ${dataStatus === "LIVE" ? "live" : ""}`} role="status">
        <span aria-hidden="true" />
        <strong>
          {dataStatus === "LIVE"
            ? "CANONICAL FEED LIVE"
            : closingAuctionActive
              ? "CLOSING AUCTION · STANDARD SIGNAL DISABLED"
              : "LIVE ANALYSIS UNAVAILABLE"}
        </strong>
        <p>
          {dataStatus === "LIVE"
            ? "Chart and analysis are bound to the same immutable snapshot."
            : `${humanizeReason(marketReason)} · BUY/SELL disabled`}
        </p>
        <time>{clockIso ? formatIstWithSeconds(clockIso) : "--:--:--"} IST</time>
      </div>

      <section className="terminal-workspace">
        <article className="chart-panel">
          <header className="panel-toolbar chart-toolbar">
            <div>
              <div className="chart-title-line">
                <p className="eyebrow">Canonical market state</p>
                <span className={`market-phase phase-${dataStatus.toLowerCase()}`}>
                  {marketPhaseLabel}
                </span>
              </div>
              <h2>NIFTY 50 <span>·</span> {timeframe.toUpperCase()}</h2>
              <OhlcLine candle={displayedQuote} />
            </div>
            <div className="indicator-control" aria-label="Chart overlays">
              <IndicatorToggle
                active={indicators.ema20}
                label="EMA 20"
                onClick={() => setIndicators((state) => ({ ...state, ema20: !state.ema20 }))}
                tone="cyan"
              />
              <IndicatorToggle
                active={indicators.ema50}
                label="EMA 50"
                onClick={() => setIndicators((state) => ({ ...state, ema50: !state.ema50 }))}
                tone="violet"
              />
              <IndicatorToggle
                active={indicators.levels}
                label="Levels"
                onClick={() => setIndicators((state) => ({ ...state, levels: !state.levels }))}
              />
              <IndicatorToggle
                active={indicators.signals}
                label="Signals"
                onClick={() => setIndicators((state) => ({ ...state, signals: !state.signals }))}
              />
            </div>
          </header>
          <TerminalChart
            analysis={analysis}
            priceAction={timeframe === "5m" ? priceAction : null}
            developing={developing}
            finalized={displayedCandles}
            indicators={indicators}
          />
          {!view && (
            <div className="chart-empty" role="status">
              <span className="empty-pulse" aria-hidden="true" />
              <strong>Waiting for the canonical feed</strong>
              <p>
                Connect a licensed provider to render NIFTY 50. No prices are being fabricated.
                No chart data is simulated.
              </p>
              <div className="empty-sequence" aria-hidden="true">
                <i /><i /><i /><i /><i />
              </div>
            </div>
          )}
          {view && dataStatus !== "LIVE" && (
            <div className="chart-stale-shield">
              <strong>{marketPhaseLabel}</strong>
              <span>
                {closingAuctionActive
                  ? "Last continuous candle retained · auction values excluded from standard signals"
                  : "Last valid chart retained · new signals disabled"}
              </span>
            </div>
          )}
          <footer className="chart-footer" data-library="lightweight-charts">
            <span>Crosshair</span>
            <span>Scroll to zoom</span>
            <span>Drag to pan</span>
            <span>Developing candle: visual only</span>
            <span className="volume-note">NIFTY spot volume unavailable</span>
            <a href="https://www.tradingview.com/" rel="noreferrer" target="_blank">
              Charts by TradingView
            </a>
          </footer>
        </article>

        <AnalysisPanel
          analysis={analysis}
          analysisReason={analysisReason}
          dataStatus={dataStatus}
          isSynced={Boolean(analysis || priceAction)}
          priceAction={priceAction}
        />
      </section>

      <MarketIntelligence analysis={analysis} dataStatus={dataStatus} priceAction={priceAction} />

      <section className="audit-bar" aria-label="Point-in-time audit trail">
        <div><span>Snapshot</span><strong>{snapshotShort}</strong></div>
        <div><span>Candle revision</span><strong>{revisionShort}</strong></div>
        <div><span>Feature set</span><strong>{PROJECT.featureVersion}</strong></div>
        <div><span>Price action</span><strong>{PROJECT.priceActionVersion}</strong></div>
        <div><span>Outcome label</span><strong>{PROJECT.labelVersion}</strong></div>
        <div><span>Calibration</span><strong>{analysis?.calibration_version ?? "NO APPROVED CALIBRATION"}</strong></div>
        <div><span>Model</span><strong>{analysis?.model_version ?? "NOT APPROVED"}</strong></div>
        <div><span>Data age</span><strong>{dataAge}</strong></div>
      </section>

      <TrackingConsole apiBaseUrl={apiBaseUrl} />

      <details className="development-progress">
        <summary>
          <span>Stage 4 research</span>
          <strong>Step {PROJECT.currentStep} of {STAGE_STEPS.length}</strong>
        </summary>
        <ol>
          {STAGE_STEPS.map((step) => (
            <li className={step.number <= PROJECT.currentStep ? "complete" : ""} key={step.number}>
              <span>{String(step.number).padStart(2, "0")}</span>
              {step.title}
            </li>
          ))}
        </ol>
      </details>
    </main>
  );
}

function subscribeToNoUpdates() {
  return () => {};
}

function useMarketApiBaseUrl(): string {
  return useSyncExternalStore(
    subscribeToNoUpdates,
    // This runs after hydration in the browser. It safely selects the
    // same-origin Worker BFF when a local development URL is not configured.
    () => CONFIGURED_API_BASE_URL || window.location.origin,
    // The server snapshot stays stable through hydration.
    () => CONFIGURED_API_BASE_URL,
  );
}

function AnalysisPanel({
  analysis,
  analysisReason,
  dataStatus,
  isSynced,
  priceAction,
}: {
  analysis: AnalysisViewContract | null;
  analysisReason: string;
  dataStatus: DataStatus;
  isSynced: boolean;
  priceAction: PriceActionAnalysisContract | null;
}) {
  const signal = analysis?.signal ?? null;
  const direction = signal?.direction ?? "WAIT";
  const directionalProbability = probabilityForDirection(signal);
  const supportiveReasons = analysis?.reasons.length
    ? analysis.reasons
    : priceAction?.reasons ?? [];
  const contradictoryReasons = analysis?.contradictory_evidence.length
    ? analysis.contradictory_evidence
    : priceAction?.contradictory_evidence ?? [];
  return (
    <aside className={`analysis-panel signal-${direction.toLowerCase()}`} aria-label="AI analysis">
      <header className="analysis-header">
        <div><p className="eyebrow">Quantitative analysis</p><h2>Decision engine</h2></div>
        <span className={`sync-badge ${isSynced ? "synced" : ""}`}>
          {isSynced ? "SNAPSHOT SYNCED" : "SYNCING ANALYSIS"}
        </span>
      </header>

      <section className="signal-card">
        <div className="signal-heading">
          <span className="signal-icon" aria-hidden="true">
            {direction === "BUY" ? "↗" : direction === "SELL" ? "↘" : "–"}
          </span>
          <div><small>Official signal</small><strong>{direction}</strong></div>
          <div className="probability-lockup">
            <small>Calibrated probability</small><strong>{directionalProbability}</strong>
          </div>
        </div>
        <div className="signal-meta">
          <span>{signal?.lifecycle_status ?? "NO SIGNAL"}</span>
          <span>{signal ? `Expires ${formatIst(signal.expires_at)}` : "60m horizon"}</span>
          <span>{dataStatus === "LIVE" ? "Data fresh" : "Signal disabled"}</span>
        </div>
        {!analysis && (
          <p className="signal-explanation">
            No calibrated probability or trade level is available. {humanizeReason(analysisReason)}.
          </p>
        )}
      </section>

      <PriceActionCard analysis={priceAction} />

      <ProbabilityMatrix signal={signal} />
      <RiskMatrix signal={signal} priceActionPlan={priceAction?.trade_plan ?? null} />

      <section className="evidence-section">
        <div className="evidence-column">
          <p className="eyebrow">Why this decision</p>
          {supportiveReasons.length ? (
            <ul>{supportiveReasons.slice(0, 5).map((reason) => <li key={reason}><i>✓</i>{reason}</li>)}</ul>
          ) : <EmptyInline text="Evidence unavailable until synchronized inference completes." />}
        </div>
        <div className="evidence-column contradictory">
          <p className="eyebrow">Contradictory evidence</p>
          {contradictoryReasons.length ? (
            <ul>{contradictoryReasons.slice(0, 5).map((reason) => <li key={reason}><i>!</i>{reason}</li>)}</ul>
          ) : <EmptyInline text="No evaluated contradictory evidence available." />}
        </div>
      </section>

      <section className="invalidation-row">
        <span>Invalidation</span>
        <strong>{signal?.risk_levels || priceAction?.trade_plan ? formatPrice(signal?.risk_levels?.invalidation ?? priceAction!.trade_plan!.invalidation) : "—"}</strong>
        <small>{signal?.risk_levels ? "Official underlying spot level" : priceAction?.trade_plan ? "Conditional price-action level" : "No active setup"}</small>
      </section>
    </aside>
  );
}

function PriceActionCard({ analysis }: { analysis: PriceActionAnalysisContract | null }) {
  const direction = analysis?.trade_plan?.direction ?? null;
  const tone = direction?.toLowerCase() ?? "wait";
  return (
    <section className={`price-action-card price-action-${tone}`} aria-label="Price action setup">
      <div className="price-action-heading">
        <div>
          <small>Conditional price action</small>
          <strong>{analysis ? humanizeReason(analysis.setup) : "CALCULATING"}</strong>
        </div>
        <div>
          <small>Confluence</small>
          <strong>{analysis ? `${analysis.confluence_score > 0 ? "+" : ""}${analysis.confluence_score} · ${analysis.evidence_grade}` : "—"}</strong>
        </div>
      </div>
      <div className="price-action-state">
        <span>Structure <strong>{analysis ? humanizeReason(analysis.structure_5m) : "—"}</strong></span>
        <span>5m <strong>{analysis?.trend_5m ?? "—"}</strong></span>
        <span>15m <strong>{analysis?.trend_15m ?? "—"}</strong></span>
        <span>1H <strong>{analysis?.trend_1h ?? "—"}</strong></span>
      </div>
      <p>
        {analysis?.blockers.length
          ? `Watch only · ${humanizeReason(analysis.blockers[0])}`
          : "Finalized-candle setup only · not a calibrated model signal"}
      </p>
    </section>
  );
}

function ProbabilityMatrix({ signal }: { signal: SignalDecisionContract | null }) {
  const probabilities = signal?.probabilities;
  return (
    <section className="probability-matrix" aria-label="Calibrated outcome probabilities">
      <ProbabilityRow label="UP" value={probabilities?.UP ?? null} tone="up" />
      <ProbabilityRow label="DOWN" value={probabilities?.DOWN ?? null} tone="down" />
      <ProbabilityRow label="NEITHER" value={probabilities?.NEITHER ?? null} tone="neutral" />
    </section>
  );
}

function ProbabilityRow({ label, value, tone }: { label: string; value: number | null; tone: string }) {
  const percent = value === null ? 0 : Math.max(0, Math.min(100, value * 100));
  return (
    <div className={`probability-row probability-${tone}`}>
      <span>{label}</span><div><i style={{ width: `${percent}%` }} /></div>
      <strong>{value === null ? "—" : `${Math.round(percent)}%`}</strong>
    </div>
  );
}

function RiskMatrix({ signal, priceActionPlan }: { signal: SignalDecisionContract | null; priceActionPlan: ConditionalTradePlanContract | null }) {
  const levels = signal?.risk_levels ?? priceActionPlan;
  return (
    <section className="risk-matrix" aria-label="Entry stop and targets">
      <div className="risk-entry"><span>{signal?.risk_levels ? "Entry zone" : "Conditional entry"}</span><strong>{levels ? `${formatPrice(levels.entry_low)} – ${formatPrice(levels.entry_high)}` : "—"}</strong></div>
      <div className="risk-stop"><span>Stop</span><strong>{levels ? formatPrice(levels.stop) : "—"}</strong></div>
      <div><span>Target 1</span><strong>{levels ? formatPrice(levels.target1) : "—"}</strong></div>
      <div><span>Target 2</span><strong>{levels ? formatPrice(levels.target2) : "—"}</strong></div>
      <div><span>Target 3</span><strong>{levels ? formatPrice(levels.target3) : "—"}</strong></div>
      <div><span>Target 1 / risk</span><strong>{levels ? `${levels.target1_reward_risk.toFixed(2)}R` : "—"}</strong></div>
    </section>
  );
}

function MarketIntelligence({ analysis, dataStatus, priceAction }: { analysis: AnalysisViewContract | null; dataStatus: DataStatus; priceAction: PriceActionAnalysisContract | null }) {
  const context = analysis?.market_context;
  return (
    <section className="intelligence-grid" aria-label="Market intelligence context">
      <article className="context-panel market-state-panel">
        <header><div><p className="eyebrow">Multi-timeframe state</p><h2>Market context</h2></div><span>{context?.status ?? "UNAVAILABLE"}</span></header>
        <div className="context-metrics">
          <ContextMetric label="Regime" value={context?.regime ?? priceAction?.trend_1h} timeframe="1H" />
          <ContextMetric label="Trend" value={context?.trend ?? priceAction?.trend_15m} timeframe="15M" />
          <ContextMetric label="Structure" value={priceAction?.structure_5m} timeframe="5M" />
          <ContextMetric label="Volatility" value={context?.volatility ?? priceAction?.volatility_regime} timeframe="5M" />
        </div>
      </article>

      <article className="context-panel news-panel">
        <header><div><p className="eyebrow">Timestamped context</p><h2>News & events</h2></div><span>{analysis?.news.status ?? "UNAVAILABLE"}</span></header>
        {analysis?.news.items.length ? (
          <ul className="news-list">
            {analysis.news.items.slice(0, 3).map((item) => (
              <li key={item.event_id}><time>{formatIst(item.published_at)}</time><div><strong>{item.headline}</strong><span>{item.source} · {item.impact}</span></div></li>
            ))}
          </ul>
        ) : <ContextEmpty title="No verified news feed" text="Headlines will appear only with source and arrival timestamps." />}
      </article>

      <article className="context-panel analog-panel">
        <header><div><p className="eyebrow">Out-of-sample references</p><h2>Historical analogs</h2></div><span>{analysis?.historical_analogs.count ?? "—"}</span></header>
        {analysis?.historical_analogs.summary ? <p className="analog-summary">{analysis.historical_analogs.summary}</p> : <ContextEmpty title="No analog set available" text="Similarity results require an approved feature snapshot and indexed history." />}
      </article>

      <article className="context-panel quality-panel">
        <header><div><p className="eyebrow">Fail-safe controls</p><h2>Data integrity</h2></div><span className={dataStatus === "LIVE" ? "ok" : "warn"}>{dataStatus}</span></header>
        <ul className="quality-list">
          <li><span>Chart / analysis snapshot</span><strong>{analysis || priceAction ? "MATCHED" : "NOT READY"}</strong></li>
          <li><span>Finalized 5m input</span><strong>{priceAction ? "VERIFIED" : "NOT READY"}</strong></li>
          <li><span>15m / 1H context</span><strong>{priceAction && priceAction.trend_15m !== "UNAVAILABLE" && priceAction.trend_1h !== "UNAVAILABLE" ? "FINALIZED" : "NOT READY"}</strong></li>
          <li><span>NIFTY spot volume</span><strong>UNAVAILABLE</strong></li>
        </ul>
      </article>
    </section>
  );
}

function ContextMetric({ label, value, timeframe }: { label: string; value: string | null | undefined; timeframe: string }) {
  return <div><span>{label}<i>{timeframe}</i></span><strong>{value ?? "Unavailable"}</strong></div>;
}

function ContextEmpty({ title, text }: { title: string; text: string }) {
  return <div className="context-empty"><i aria-hidden="true">·</i><div><strong>{title}</strong><p>{text}</p></div></div>;
}

function EmptyInline({ text }: { text: string }) {
  return <p className="empty-inline">{text}</p>;
}

function OhlcLine({ candle }: { candle: CandleContract | null }) {
  return (
    <div className="ohlc-line" aria-label="Latest candle OHLC">
      <span>O <strong>{candle ? formatPrice(candle.open) : "—"}</strong></span>
      <span>H <strong>{candle ? formatPrice(candle.high) : "—"}</strong></span>
      <span>L <strong>{candle ? formatPrice(candle.low) : "—"}</strong></span>
      <span>C <strong>{candle ? formatPrice(candle.close) : "—"}</strong></span>
      <span>V <strong>N/A</strong></span>
    </div>
  );
}

function IndicatorToggle({ active, label, onClick, tone = "default" }: { active: boolean; label: string; onClick: () => void; tone?: string }) {
  return <button aria-pressed={active} className={`${active ? "active" : ""} tone-${tone}`} onClick={onClick} type="button"><i aria-hidden="true" />{label}</button>;
}

function DataStatusBadge({ status, age, label }: { status: DataStatus; age: string; label: string }) {
  return <div className={`data-status status-${status.toLowerCase()}`}><span aria-hidden="true" /><div><small>Market data</small><strong>{label}</strong><em>Age {age}</em></div></div>;
}

function synchronizedAnalysis(view: MarketStateViewContract | null, availability: AnalysisAvailabilityContract | null, dataStatus: DataStatus): AnalysisViewContract | null {
  if (!view || !availability || dataStatus !== "LIVE") return null;
  if (!view.snapshot.live_inference_eligible) return null;
  if (availability.sync_state !== "SYNCED" || !availability.analysis) return null;
  if (availability.snapshot_id !== view.snapshot.snapshot_id) return null;
  if (availability.candle_revision_checksum !== view.snapshot.candle_revision_checksum) return null;
  return availability.analysis;
}

function synchronizedPriceAction(
  view: MarketStateViewContract | null,
  availability: PriceActionAvailabilityContract | null,
): PriceActionAnalysisContract | null {
  if (!view || !availability || availability.sync_state !== "SYNCED" || !availability.analysis) return null;
  if (availability.snapshot_id !== view.snapshot.snapshot_id) return null;
  if (availability.candle_revision_checksum !== view.snapshot.candle_revision_checksum) return null;
  return availability.analysis;
}

function probabilityForDirection(signal: SignalDecisionContract | null) {
  if (!signal?.probabilities) return "—";
  const probability = signal.direction === "BUY"
    ? signal.probabilities.UP
    : signal.direction === "SELL"
      ? signal.probabilities.DOWN
      : Math.max(signal.probabilities.UP, signal.probabilities.DOWN, signal.probabilities.NEITHER);
  return `${Math.round(probability * 100)}%`;
}

function priceChange(latest: CandleContract | null, previous: CandleContract | null) {
  if (!latest || !previous) return { label: "—", direction: "flat" };
  const current = Number(latest.close);
  const prior = Number(previous.close);
  const delta = current - prior;
  const percent = prior ? (delta / prior) * 100 : 0;
  return { label: `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} (${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%)`, direction: delta > 0 ? "positive" : delta < 0 ? "negative" : "flat" };
}

function formatPrice(value: string) {
  return new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(Number(value));
}

function formatIst(value: string) {
  return new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(value));
}

function formatIstWithSeconds(value: string) {
  return new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(value));
}

function formatAge(value: string) {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3_600) return `${Math.floor(seconds / 60)}m`;
  return `${Math.floor(seconds / 3_600)}h`;
}

function humanizeReason(value: string) {
  return value.toLowerCase().replaceAll("_", " ");
}

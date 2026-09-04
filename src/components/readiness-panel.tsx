"use client";

import { useEffect, useState } from "react";

import { fetchReleaseReadiness } from "@/src/lib/hardening-client";
import type { ReleaseReadinessContract } from "@/src/lib/hardening-contracts";

export function ReadinessPanel({ apiBaseUrl }: { apiBaseUrl: string }) {
  const [readiness, setReadiness] = useState<ReleaseReadinessContract | null>(null);
  const [reason, setReason] = useState(apiBaseUrl ? "CHECKING_RELEASE_GATES" : "MARKET_API_NOT_CONFIGURED");

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
      const next = await fetchReleaseReadiness(apiBaseUrl, controller.signal);
      if (stopped) return;
      setReadiness(next);
      setReason(next ? "RELEASE_GATE_EVALUATED" : "READINESS_API_UNAVAILABLE");
      timer = setTimeout(refresh, 15_000);
    };
    void refresh();
    return () => {
      stopped = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [apiBaseUrl]);

  const blockers = readiness?.blockers ?? [];
  return (
    <article className="observability-panel readiness-panel" id="readiness">
      <header className="observability-header">
        <div><p className="eyebrow">Step 10 · Production gate</p><h3>Live-signal readiness</h3></div>
        <span className={`evidence-chip evidence-${(readiness?.status ?? "UNAVAILABLE").toLowerCase()}`}>
          {readiness?.status ?? "UNAVAILABLE"}
        </span>
      </header>
      <div className="readiness-summary">
        <ReadinessMetric label="Signal permission" value={readiness?.signal_allowed ? "ALLOWED" : "BLOCKED"} />
        <ReadinessMetric label="Snapshot parity" value={readiness?.exact_snapshot_match ? "MATCHED" : "NOT READY"} />
        <ReadinessMetric label="API security" value={readiness?.security_mode ?? "UNAVAILABLE"} />
        <ReadinessMetric label="Kill switch" value={readiness?.kill_switch_active ? "ACTIVE" : "CLEAR"} />
        <ReadinessMetric label="Release artifact" value={readiness?.release_id ?? "NOT APPROVED"} />
      </div>
      {blockers.length ? (
        <ul className="release-blockers">
          {blockers.slice(0, 8).map((blocker) => <li key={blocker}><i aria-hidden="true">×</i>{humanize(blocker)}</li>)}
        </ul>
      ) : (
        <p className="readiness-empty">{readiness ? "Every deterministic release gate passed." : `${humanize(reason)}. No readiness claim is being made.`}</p>
      )}
      <footer className="readiness-footer">No gate can be overridden by the dashboard or an LLM. BUY/SELL remains suppressed until this contract is READY.</footer>
    </article>
  );
}

function ReadinessMetric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function humanize(value: string) {
  return value.toLowerCase().replaceAll("_", " ");
}

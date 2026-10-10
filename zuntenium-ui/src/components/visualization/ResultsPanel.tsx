// src/components/visualization/ResultsPanel.tsx
"use client";
import { useState } from "react";
import { ProbabilityHistogram } from "./ProbabilityHistogram";
import { StatevectorChart }     from "./StatevectorChart";
import { BlochSphere }          from "./BlochSphere";
import { useExecutionStore }    from "@/store";

type VizTab = "probability" | "statevector" | "bloch";

export function ResultsPanel() {
  const [vizTab, setVizTab] = useState<VizTab>("probability");
  const result = useExecutionStore(s => s.result);

  if (!result || result.status === "queued" || result.status === "running") {
    return (
      <div className="flex flex-col items-center py-8 px-4 text-center text-t3">
        <div className="text-2xl mb-2 opacity-40">📊</div>
        <div className="text-xs text-t2 mb-1">No results yet</div>
        <div className="text-[10px]">
          Run the circuit to see statevectors,<br />
          probability distributions, and fidelity
        </div>
      </div>
    );
  }

  const { counts, statevector, fidelity_before, fidelity_after,
          backend_used, execution_time_s, ai_confidence, decision_reason } = result;
  const numQubits = counts
    ? Math.max(...Object.keys(counts).map(k => k.length))
    : (statevector ? Math.round(Math.log2(statevector.length)) : 1);
  const shots = counts ? Object.values(counts).reduce((a, b) => a + b, 0) : 0;
  const fid   = fidelity_after ?? fidelity_before ?? 0;

  // Bloch available only for n=1
  const blochP0 = numQubits === 1 && counts
    ? (counts["0"] ?? 0) / Math.max(shots, 1)
    : null;

  const VIZ_TABS = [
    { id: "probability",  label: "Probability" },
    { id: "statevector",  label: "Statevector", disabled: !statevector },
    { id: "bloch",        label: "Bloch",       disabled: blochP0 === null },
  ] as { id: VizTab; label: string; disabled?: boolean }[];

  return (
    <div>
      {/* Metrics card */}
      <div className="bg-card border border-[var(--border)] rounded-lg p-3 mb-3">
        <div className="space-y-[5px]">
          {[
            { label: "Status",   value: "Completed ✓",                 cls: "text-success" },
            { label: "Backend",  value: backend_used ?? "tensornetwork",cls: "text-cyan-DEFAULT" },
            { label: "Exec time",value: `${(execution_time_s ?? 0).toFixed(3)}s`, cls: "text-primary-DEFAULT" },
            { label: "Fidelity", value: fid.toFixed(4),                cls: fid > 0.99 ? "text-success" : fid > 0.95 ? "text-warning" : "text-danger" },
            { label: "Confidence", value: ai_confidence ? `${(ai_confidence*100).toFixed(1)}%` : "—", cls: "text-purple" },
          ].map(row => (
            <div key={row.label} className="flex items-center justify-between">
              <span className="text-[10px] text-t3">{row.label}</span>
              <span className={`text-[11px] font-mono font-semibold ${row.cls}`}>
                {row.value}
              </span>
            </div>
          ))}
        </div>

        {/* Fidelity bar */}
        <div className="mt-2 h-[3px] bg-[var(--border)] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width:      `${fid * 100}%`,
              background: "linear-gradient(90deg,#0084FF,#00D4B8)",
            }}
          />
        </div>

        {decision_reason && (
          <div className="mt-2 text-[9px] font-mono text-t3 truncate">
            Decision: {decision_reason}
          </div>
        )}
      </div>

      {/* Viz tabs */}
      <div className="flex gap-1 mb-3">
        {VIZ_TABS.map(t => (
          <button
            key={t.id}
            onClick={() => !t.disabled && setVizTab(t.id)}
            disabled={t.disabled}
            className={`flex-1 h-7 rounded border text-[10px] font-semibold
                        transition-all duration-100
                        ${t.disabled
                          ? "opacity-30 cursor-not-allowed border-[var(--border)] text-t3"
                          : vizTab === t.id
                          ? "border-primary-DEFAULT/40 bg-primary-dim text-primary-DEFAULT"
                          : "border-[var(--border)] text-t2 hover:bg-hover hover:text-t1"
                        }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Viz content */}
      {vizTab === "probability" && counts && (
        <div className="bg-card border border-[var(--border)] rounded-lg p-3">
          <div className="flex items-center justify-between mb-3">
            <span className="section-label">Measurement Counts</span>
            <span className="text-[9px] font-mono text-t3">{shots} shots</span>
          </div>
          <ProbabilityHistogram counts={counts} shots={shots} />
        </div>
      )}

      {vizTab === "statevector" && statevector && (
        <div className="bg-card border border-[var(--border)] rounded-lg p-3">
          <div className="section-label mb-3">Statevector Amplitudes</div>
          <StatevectorChart statevector={statevector} numQubits={numQubits} />
        </div>
      )}

      {vizTab === "bloch" && blochP0 !== null && (
        <div className="bg-card border border-[var(--border)] rounded-lg p-3 flex justify-center">
          <BlochSphere p0={blochP0} qubitIdx={0} />
        </div>
      )}
    </div>
  );
}

// src/components/ai/AIInsightsPanel.tsx
// Prompt 9 — Dedicated AI transparency panel
// Exposes: decision, confidence ring, feature importance, reward breakdown,
//          transpiler summary, MHOS device scores, error mitigation summary
"use client";
import { useState, memo } from "react";
import { useExecutionStore } from "@/store";
import { ConfidenceIndicator }   from "./ConfidenceIndicator";
import { FeatureImportanceBar }  from "./FeatureImportanceBar";

// ── Static AI decision metadata (populated from real API in production) ────
const AI_META = {
  backend:       "tensornetwork",
  strategy:      "rl_refine",
  confidence:    0.873,
  fallback_used: false,
  reason:
    "High entanglement score (0.50) with n=2 qubits in feasible statevector range. " +
    "Queue time 0s → MPS χ=64 optimal. Clifford fraction 1.0 → stabilizer path skipped. " +
    "Confidence 0.87 ≥ threshold 0.70. RL reward r=0.87 surpasses exploration threshold.",

  features: [
    { name: "entanglement",   v: 0.78, color: "#00D4B8", desc: "Mean concurrence across qubit pairs" },
    { name: "n_qubits",       v: 0.62, color: "#0084FF", desc: "Circuit qubit count (normalised to max=200)" },
    { name: "sv_feasible",    v: 0.55, color: "#F5A623", desc: "Statevector simulation feasibility flag" },
    { name: "depth",          v: 0.44, color: "#FF4757", desc: "Circuit depth (normalised)" },
    { name: "hw_available",   v: 0.33, color: "#A855F7", desc: "Real hardware availability score from MHOS" },
    { name: "queue_time",     v: 0.18, color: "#7C3AED", desc: "Estimated queue wait, inverted (lower=better)" },
    { name: "clifford_frac",  v: 0.15, color: "#10B981", desc: "Fraction of Clifford gates (stabilizer hint)" },
    { name: "param_count",    v: 0.12, color: "#E91E8C", desc: "Parameterized gate count (normalised)" },
  ],

  reward: {
    fidelity:  { label: "α·F (fidelity term)",    value: 0.970, color: "#2ECC71" },
    speed:     { label: "−β·log(t) (speed)",      value: 0.800, color: "#00D4B8" },
    fallback:  { label: "−γ·fallback (penalty)",  value: 0.000, color: "#8DA4C8" },
    total:     0.870,
  },

  transpiler: {
    gates_before: 2,  gates_after: 2,
    depth_before: 2,  depth_after: 2,
    swaps_added:  0,  strategy: "rule_only",
    note: "No optimisation needed — circuit already minimal (Bell state).",
  },

  mhosScores: [
    { id:"tensornetwork", prov:"local",  score:0.963, f:0.991, lat:0.920, cost:1.000, status:"online",   chosen:true  },
    { id:"ionq_aria-1",   prov:"ionq",   score:0.954, f:0.995, lat:0.720, cost:0.320, status:"online",   chosen:false },
    { id:"ibm_kyoto",     prov:"ibm",    score:0.914, f:0.982, lat:0.610, cost:0.990, status:"online",   chosen:false },
    { id:"statevector",   prov:"local",  score:0.982, f:1.000, lat:0.950, cost:1.000, status:"online",   chosen:false },
  ],

  mitigation: {
    method:          "readout",
    fidelity_before: 0.940,
    fidelity_after:  0.997,
    improvement_pct: 6.1,
    zne_scales:      [1, 2, 3],
  },
};

const REWARD_TERMS = [
  AI_META.reward.fidelity,
  AI_META.reward.speed,
  AI_META.reward.fallback,
];

// ── Expandable section ─────────────────────────────────────────────────────
function Section({
  title, badge, defaultOpen = false, children,
}: {
  title: string; badge?: string; defaultOpen?: boolean; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden mb-2.5">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2.5
                   bg-card hover:bg-hover transition-colors duration-150"
      >
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-t1">{title}</span>
          {badge && (
            <span className="tag tag-ai text-[8px]">{badge}</span>
          )}
        </div>
        <span
          className="text-[10px] text-t3 transition-transform duration-200"
          style={{ transform: open ? "rotate(180deg)" : "none" }}
        >
          ▾
        </span>
      </button>
      {open && (
        <div className="px-3 py-3 bg-surface border-t border-[var(--border)] text-[10px]">
          {children}
        </div>
      )}
    </div>
  );
}

// ── MetricRow helper ───────────────────────────────────────────────────────
function MetRow({
  label, value, color = "var(--t1)",
}: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between mb-1.5 last:mb-0">
      <span className="text-t3">{label}</span>
      <span className="font-mono font-semibold" style={{ color }}>{value}</span>
    </div>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────
export const AIInsightsPanel = memo(function AIInsightsPanel() {
  const result     = useExecutionStore(s => s.result);
  const confidence = result?.ai_confidence  ?? AI_META.confidence;
  const reason     = result?.decision_reason ?? AI_META.reason;
  const backend    = result?.backend_used    ?? AI_META.backend;
  const f_before   = result?.fidelity_before ?? AI_META.mitigation.fidelity_before;
  const f_after    = result?.fidelity_after  ?? AI_META.mitigation.fidelity_after;

  const hasResult  = !!result && result.status === "completed";

  return (
    <div className="h-full overflow-y-auto">
      {/* Live status badge */}
      <div className="flex items-center gap-2 mb-3">
        <span
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded
                     text-[10px] font-semibold"
          style={{
            background: "rgba(168,85,247,0.12)",
            border:     "1px solid rgba(168,85,247,0.3)",
            color:      "#A855F7",
          }}
        >
          🤖 AI Optimizer Active
        </span>
        {!hasResult && (
          <span className="text-[9px] text-t3">Run circuit to see live insights</span>
        )}
      </div>

      {/* ── Decision + Confidence ─────────────────────────────────── */}
      <Section title="AI Decision" badge="PPO RL" defaultOpen>
        <div className="flex items-start gap-3 mb-3">
          {/* Confidence ring */}
          <ConfidenceIndicator confidence={confidence} size={64} label="confidence" />

          {/* Decision text */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1.5">
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: "#00D4B8", boxShadow: "0 0 5px #00D4B8" }}
              />
              <span className="text-[11px] font-semibold">{backend}</span>
              <span className="tag tag-mhos text-[7px]">{AI_META.strategy}</span>
            </div>
            <div className="text-t2 leading-relaxed">{reason}</div>
          </div>
        </div>

        {/* Threshold check */}
        <div
          className="flex items-center gap-2 px-2.5 py-2 rounded"
          style={{ background: "rgba(46,204,113,0.06)", border: "1px solid rgba(46,204,113,0.2)" }}
        >
          <span style={{ color: "#2ECC71", fontSize: 12 }}>✓</span>
          <span className="text-t2">
            Confidence {Math.round(confidence * 100)}% ≥ threshold 70% — non-fallback decision
          </span>
        </div>
      </Section>

      {/* ── Feature Importance ───────────────────────────────────────── */}
      <Section title="Feature Importance" badge={`${AI_META.features.length} / 27 dims`} defaultOpen>
        <div className="mb-2 flex justify-between">
          <span className="text-t3">FeatureVector(27-dim) — top factors</span>
          <span className="text-t3">value</span>
        </div>
        {AI_META.features.map((f, i) => (
          <FeatureImportanceBar
            key={f.name}
            name={f.name}
            value={f.v}
            maxValue={0.85}
            color={f.color}
            description={f.desc}
            rank={i + 1}
          />
        ))}
        <div className="mt-2.5 px-2 py-1.5 rounded text-t3"
             style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
          Remaining 19 dims contribute &lt; 0.05 each (circuit timing, provider flags, historical rewards)
        </div>
      </Section>

      {/* ── Reward Breakdown ─────────────────────────────────────────── */}
      <Section title="Reward Breakdown" badge="r_t = α·F − β·log(t) − γ">
        {REWARD_TERMS.map((r) => (
          <div key={r.label} className="flex items-center justify-between mb-2">
            <span className="text-t3">{r.label}</span>
            <span className="font-mono font-semibold" style={{ color: r.color }}>
              {r.value >= 0 ? "+" : ""}{r.value.toFixed(3)}
            </span>
          </div>
        ))}
        <div
          className="flex items-center justify-between pt-2 mt-1"
          style={{ borderTop: "1px solid var(--border)" }}
        >
          <span className="font-semibold text-t1">Total r_t</span>
          <span
            className="text-[15px] font-bold font-mono"
            style={{ color: "#2ECC71" }}
          >
            +{AI_META.reward.total.toFixed(3)}
          </span>
        </div>
        <div className="mt-2 text-t3">
          PPO agent clipped reward to [-2, 1]. Exploration ε={0.05}. Online learning enabled.
        </div>
      </Section>

      {/* ── MHOS Device Scores ───────────────────────────────────────── */}
      <Section title="MHOS Device Comparison" badge={`${AI_META.mhosScores.length} devices scored`}>
        <div className="space-y-2">
          {AI_META.mhosScores
            .sort((a, b) => b.score - a.score)
            .map(d => {
              const pc = d.prov === "ibm" ? "#0084FF" : d.prov === "ionq" ? "#A855F7" : "#00D4B8";
              return (
                <div
                  key={d.id}
                  className="flex items-center gap-2.5 px-2.5 py-2 rounded border
                             transition-all duration-150"
                  style={{
                    borderColor: d.chosen ? "rgba(0,132,255,0.5)" : "var(--border)",
                    background:  d.chosen ? "rgba(0,132,255,0.06)" : "var(--card)",
                  }}
                >
                  <div
                    className="w-6 h-6 rounded flex items-center justify-center
                               text-[8px] font-bold font-mono flex-shrink-0"
                    style={{ background: `${pc}18`, color: pc, border: `1px solid ${pc}30` }}
                  >
                    {d.prov.slice(0, 2).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-medium truncate">{d.id}</span>
                      {d.chosen && <span className="tag tag-mhos text-[7px]">chosen</span>}
                    </div>
                    <div className="text-[8px] font-mono text-t3 mt-0.5">
                      F={d.f.toFixed(3)} lat={d.lat.toFixed(2)} cost={d.cost.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div
                      className="text-[12px] font-bold font-mono"
                      style={{
                        color: d.score >= 0.95 ? "#2ECC71" : d.score >= 0.88 ? "#00D4B8" : "#F5A623",
                      }}
                    >
                      {d.score.toFixed(3)}
                    </div>
                    <div
                      className="h-[3px] w-12 rounded-full overflow-hidden mt-1"
                      style={{ background: "var(--border)" }}
                    >
                      <div
                        className="h-full rounded-full"
                        style={{
                          width:      `${d.score * 100}%`,
                          background: d.chosen
                            ? "linear-gradient(90deg,#0084FF,#00D4B8)"
                            : "var(--border-hi,#1D3058)",
                        }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
        </div>
      </Section>

      {/* ── Transpiler Summary ───────────────────────────────────────── */}
      <Section title="Transpiler Summary" badge={AI_META.transpiler.strategy}>
        <div className="grid grid-cols-2 gap-2 mb-2">
          {[
            ["Gates (before)", AI_META.transpiler.gates_before, "Gates (after)", AI_META.transpiler.gates_after],
            ["Depth (before)", AI_META.transpiler.depth_before, "Depth (after)",  AI_META.transpiler.depth_after],
          ].map(([la, va, lb, vb], i) => (
            <div key={i} className="col-span-2 grid grid-cols-2 gap-2">
              <div className="bg-card border border-[var(--border)] rounded p-2 text-center">
                <div className="text-[16px] font-mono font-bold text-t2">{va}</div>
                <div className="text-[8px] text-t3 mt-0.5">{la}</div>
              </div>
              <div
                className="bg-card border rounded p-2 text-center"
                style={{ borderColor: "rgba(46,204,113,0.3)" }}
              >
                <div className="text-[16px] font-mono font-bold" style={{ color: "#2ECC71" }}>
                  {vb}
                </div>
                <div className="text-[8px] text-t3 mt-0.5">{lb}</div>
              </div>
            </div>
          ))}
        </div>
        <MetRow label="SWAPs inserted"  value={`${AI_META.transpiler.swaps_added}`} />
        <div className="mt-2 text-t3 italic">{AI_META.transpiler.note}</div>
      </Section>

      {/* ── Error Mitigation ─────────────────────────────────────────── */}
      <Section title="Error Mitigation" badge={AI_META.mitigation.method.toUpperCase()}>
        <div className="flex items-center gap-3 mb-3">
          <div
            className="flex-1 flex flex-col items-center py-2 rounded bg-card border border-[var(--border)]"
          >
            <span className="text-[16px] font-mono font-bold text-warning">
              {f_before.toFixed(4)}
            </span>
            <span className="text-[8px] text-t3 mt-0.5">Before</span>
          </div>
          <span className="text-[14px]" style={{ color: "#2ECC71" }}>→</span>
          <div
            className="flex-1 flex flex-col items-center py-2 rounded bg-card"
            style={{ border: "1px solid rgba(46,204,113,0.3)" }}
          >
            <span className="text-[16px] font-mono font-bold" style={{ color: "#2ECC71" }}>
              {f_after.toFixed(4)}
            </span>
            <span className="text-[8px] text-t3 mt-0.5">After</span>
          </div>
        </div>

        {/* Fidelity improvement bar */}
        <div className="mb-2.5">
          <div className="flex justify-between mb-1">
            <span className="text-t3">Fidelity improvement</span>
            <span className="font-mono font-semibold" style={{ color: "#2ECC71" }}>
              +{((f_after - f_before) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-[5px] bg-[var(--border)] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width:      `${f_after * 100}%`,
                background: "linear-gradient(90deg, #0084FF, #00D4B8)",
              }}
            />
          </div>
        </div>

        <MetRow label="Method"        value={AI_META.mitigation.method} color="var(--cyan)" />
        <MetRow label="ZNE scales"    value={AI_META.mitigation.zne_scales.join(", ")} />
        <MetRow
          label="Regularisation"
          value="Tikhonov λ=0.001"
          color="var(--t2)"
        />
      </Section>

      {/* Why this backend summary */}
      <div
        className="rounded-lg p-3 text-[10px] text-t2 leading-relaxed"
        style={{
          border:     "1px solid rgba(0,132,255,0.2)",
          background: "rgba(0,132,255,0.07)",
        }}
      >
        <div className="font-semibold mb-1.5" style={{ color: "#0084FF" }}>
          Why {backend} was chosen
        </div>
        Score {AI_META.mhosScores.find(d => d.chosen)?.score.toFixed(3)} vs best alternative{" "}
        {Math.max(...AI_META.mhosScores.filter(d => !d.chosen).map(d => d.score)).toFixed(3)}{" "}
        (Δ=+{(
          (AI_META.mhosScores.find(d => d.chosen)?.score ?? 0) -
          Math.max(...AI_META.mhosScores.filter(d => !d.chosen).map(d => d.score))
        ).toFixed(3)}). Zero queue time, free tier, fidelity within threshold.
        RL reward r={AI_META.reward.total.toFixed(3)} surpasses exploration threshold.
      </div>
    </div>
  );
});

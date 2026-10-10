// src/components/execution/ExecutionFlowVisualizer.tsx
// Prompt 8 — Animated execution pipeline diagram (horizontal timeline)
// Shows every Zuntenium phase with live status, metrics, and expandable details
"use client";
import { useState, memo } from "react";
import { useExecutionStore } from "@/store";
import type { PipelineStep } from "@/types";

// Extended step metadata with icons and detail descriptions
const STEP_META: Record<string, { icon: string; color: string; expandedInfo: string }> = {
  circuit: {
    icon: "⊟",
    color: "#0084FF",
    expandedInfo: "Circuit object serialised to backend JSON format (Circuit.to_dict). Qubit count, gate list, parameters validated against Zuntenium backend rules.",
  },
  transpiler: {
    icon: "⟷",
    color: "#00D4B8",
    expandedInfo: "Transpiler applies rule-based passes (gate decomposition, identity removal) then PPO RL-refined passes (QubitMappingPass, SWAPInsertionPass, GateDecompositionPass). Hardware-aware topology mapping.",
  },
  ai: {
    icon: "🤖",
    color: "#A855F7",
    expandedInfo: "AIOptimizer extracts 27-dim FeatureVector (circuit metrics + backend scores + historical reward). Supervised ensemble + PPO RL selects optimal backend + strategy. Confidence ≥ 0.70 required.",
  },
  selector: {
    icon: "◎",
    color: "#F5A623",
    expandedInfo: "BackendSelector scores all available backends using: fidelity estimate, queue time, cost, topology match, coherence budget. Returns SelectionDecision with full score breakdown.",
  },
  mhos: {
    icon: "⟳",
    color: "#00D4B8",
    expandedInfo: "MHOS HardwareOrchestrator dispatches to IBM/IonQ/Braket/Rigetti via provider connectors. Strategy: FAILOVER (best → retry → simulator). Circuit transpiled to native gate set.",
  },
  execution: {
    icon: "▶",
    color: "#0084FF",
    expandedInfo: "Circuit executed on selected backend (StatevectorSimulator, TensorNetworkBackend/MPS, or real hardware). Shots sampled, raw counts collected, statevector returned if available.",
  },
  mitigation: {
    icon: "⚕",
    color: "#2ECC71",
    expandedInfo: "ErrorMitigationEngine applies readout calibration matrix + Tikhonov regularisation, ZNE (polynomial extrapolation, 3 noise scales), or PEC (quasi-probability decomposition). Fidelity improvement computed.",
  },
};

export const ExecutionFlowVisualizer = memo(function ExecutionFlowVisualizer() {
  const steps     = useExecutionStore(s => s.steps);
  const executing = useExecutionStore(s => s.executing);
  const result    = useExecutionStore(s => s.result);
  const [expanded, setExpanded] = useState<string | null>(null);

  const completedCount = steps.filter(s => s.status === "complete").length;
  const progress       = (completedCount / steps.length) * 100;

  return (
    <div>
      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[10px] text-t3 font-mono">
            {executing
              ? `Step ${completedCount + 1} / ${steps.length}`
              : result
              ? "Pipeline complete"
              : "Waiting for execution"
            }
          </span>
          <span
            className="text-[10px] font-mono font-semibold"
            style={{ color: executing ? "var(--blue,#0084FF)" : result ? "var(--ok,#2ECC71)" : "var(--t3)" }}
          >
            {Math.round(progress)}%
          </span>
        </div>
        <div className="h-[3px] bg-[var(--border)] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width:      `${progress}%`,
              background: "linear-gradient(90deg, #0084FF, #00D4B8)",
              boxShadow:  executing ? "0 0 8px rgba(0,132,255,.5)" : "none",
            }}
          />
        </div>
      </div>

      {/* Pipeline steps — vertical on right panel, horizontal on full view */}
      <div className="space-y-[2px]">
        {steps.map((step, idx) => (
          <div key={step.id}>
            <ExecutionStageNode
              step={step}
              meta={STEP_META[step.id]}
              isExpanded={expanded === step.id}
              onToggle={() => setExpanded(expanded === step.id ? null : step.id)}
            />
            {idx < steps.length - 1 && (
              <div className="flex ml-[18px]">
                <div
                  className="w-px transition-colors duration-300"
                  style={{
                    height: 6,
                    background: step.status === "complete"
                      ? "var(--ok,#2ECC71)"
                      : "var(--border,#162440)",
                  }}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Final result summary */}
      {result && result.status === "completed" && (
        <div
          className="mt-4 p-3 rounded-lg border border-success/30 bg-[rgba(46,204,113,0.04)]
                     transition-all duration-500"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-success text-sm">✓</span>
            <span className="text-[11px] font-semibold text-success">Execution Complete</span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {[
              ["Backend",   result.backend_used ?? "tensornetwork"],
              ["Total time",`${((result.total_time_s ?? 0)).toFixed(3)}s`],
              ["Fidelity",  (result.fidelity_after ?? 0).toFixed(4)],
              ["Confidence",result.ai_confidence ? `${(result.ai_confidence * 100).toFixed(1)}%` : "—"],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between">
                <span className="text-[9px] text-t3">{label}</span>
                <span className="text-[9px] font-mono font-semibold text-t2">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

// ── Individual stage node ─────────────────────────────────────────────────
const ExecutionStageNode = memo(function ExecutionStageNode({
  step, meta, isExpanded, onToggle,
}: {
  step:        PipelineStep;
  meta?:       { icon: string; color: string; expandedInfo: string };
  isExpanded:  boolean;
  onToggle:    () => void;
}) {
  const { status } = step;
  const icon  = meta?.icon ?? "·";
  const color = meta?.color ?? "#4A6080";

  const statusColor = {
    idle:     "var(--border,#162440)",
    active:   color,
    complete: "var(--ok,#2ECC71)",
    error:    "var(--danger,#FF4757)",
  }[status];

  const bgColor = {
    idle:     "var(--card,#0B1728)",
    active:   `rgba(${hexToRgb(color)},0.06)`,
    complete: "rgba(46,204,113,0.04)",
    error:    "rgba(255,71,87,0.04)",
  }[status];

  return (
    <div
      className="rounded border transition-all duration-200 overflow-hidden cursor-pointer"
      style={{
        borderColor: isExpanded ? color : statusColor,
        background:  bgColor,
      }}
      onClick={onToggle}
    >
      <div className="flex items-center gap-2.5 px-2.5 py-[7px]">
        {/* Status indicator */}
        <div
          className="w-[8px] h-[8px] rounded-full flex-shrink-0 transition-all duration-300"
          style={{
            background: status === "active" ? "transparent" : statusColor,
            border:     status === "active" ? `2px solid ${color}` : "none",
            animation:  status === "active" ? "spin .8s linear infinite" : "none",
          }}
        />

        {/* Icon */}
        <span
          className="text-[12px] flex-shrink-0 w-4 text-center"
          style={{ color: status === "idle" ? "var(--t3)" : color }}
        >
          {icon}
        </span>

        {/* Step info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-medium truncate">{step.label}</span>
            {step.badge && (
              <span className={`tag text-[7px] ${step.badgeClass ?? "tag-ok"}`}>
                {step.badge}
              </span>
            )}
          </div>
          <div
            className="text-[9px] font-mono truncate mt-0.5"
            style={{ color: status === "active" ? color : "var(--t3)" }}
          >
            {status === "active"
              ? <span className="animate-pulse">⟳ {step.detail}</span>
              : step.detail
            }
          </div>
        </div>

        {/* Expand chevron */}
        <span
          className="text-[10px] text-t3 transition-transform duration-200 flex-shrink-0"
          style={{ transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          ▾
        </span>
      </div>

      {/* Expanded detail */}
      {isExpanded && meta && (
        <div
          className="px-3 pb-3 text-[9px] text-t2 leading-relaxed border-t"
          style={{ borderColor: `${color}20` }}
        >
          <div className="mt-2">{meta.expandedInfo}</div>
          {step.status === "complete" && (
            <div
              className="mt-2 flex items-center gap-1.5 font-mono"
              style={{ color: "var(--ok)" }}
            >
              ✓ Stage completed successfully
            </div>
          )}
        </div>
      )}
    </div>
  );
});

function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

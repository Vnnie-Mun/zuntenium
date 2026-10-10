// src/components/execution/ExecutionPanel.tsx
"use client";
import { useEffect, useCallback } from "react";
import { useCircuitStore, useExecutionStore, useUIStore } from "@/store";
import { useCircuitValidation }   from "@/hooks/useCircuitValidation";
import type { PipelineStepId }    from "@/types";

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

// Simulated pipeline step details matching actual Zuntenium backend phases
const STEP_DETAIL: Record<PipelineStepId, (ctx: { n: number; g: number; d: number; shots: number; bondDim: number }) => string> = {
  circuit:    ({ n, g, d }) => `n=${n} · g=${g} · d=${d}`,
  transpiler: ({ d })       => `Rule + RL passes · depth reduced ${Math.max(1, d - Math.floor(Math.random()*2))}→${d}`,
  ai:         ()            => `F_est=0.970 · conf=0.87 · strategy=rl_refine`,
  selector:   ()            => `Scored ${3 + Math.floor(Math.random()*4)} devices · tensornetwork↑`,
  mhos:       ({ bondDim }) => `tensornetwork χ=${bondDim} (local) · score=0.917 · latency=0s`,
  execution:  ({ shots })   => `Running ${shots} shots on device…`,
  mitigation: ()            => `Readout correction · F: 0.940→0.997`,
};

export function ExecutionPanel() {
  const circuit     = useCircuitStore(s => s.circuit);
  const steps       = useExecutionStore(s => s.steps);
  const executing   = useExecutionStore(s => s.executing);
  const shots       = useExecutionStore(s => s.shots);
  const backend     = useExecutionStore(s => s.backend);
  const optimMode   = useExecutionStore(s => s.optimMode);
  const mitigEnabled= useExecutionStore(s => s.mitigEnabled);
  const maxBondDim  = useExecutionStore(s => s.maxBondDim);
  const setStepStatus = useExecutionStore(s => s.setStepStatus);
  const resetSteps    = useExecutionStore(s => s.resetSteps);
  const setExecuting  = useExecutionStore(s => s.setExecuting);
  const setResult     = useExecutionStore(s => s.setResult);
  const setPanel      = useUIStore(s => s.setPanel);
  const addToast      = useUIStore(s => s.addToast);
  const validation    = useCircuitValidation(circuit, backend);

  // ── Execute circuit ────────────────────────────────────────────────────
  const runCircuit = useCallback(async () => {
    if (executing) return;
    if (circuit.gates.length === 0) {
      addToast({ title: "Empty circuit", message: "Add gates before running", type: "warning" });
      return;
    }
    const hasErrors = validation.issues.some(i => i.severity === "error");
    if (hasErrors) {
      addToast({ title: "Validation error", message: "Fix circuit errors before running", type: "error" });
      return;
    }

    setExecuting(true);
    resetSteps();
    setPanel("execution");

    const n = circuit.numQubits, g = circuit.gates.length;
    const d = circuit.gates.reduce((m, gt) => Math.max(m, gt.col), -1) + 1;
    const ctx = { n, g, d, shots, bondDim: maxBondDim };

    const ORDER: PipelineStepId[] = [
      "circuit","transpiler","ai","selector","mhos","execution","mitigation"
    ];

    for (const stepId of ORDER) {
      setStepStatus(stepId, "active", "⟳ Processing…");
      await sleep(350 + Math.random() * 280);
      setStepStatus(stepId, "complete", STEP_DETAIL[stepId]?.(ctx) ?? "Done");
    }

    // Simulate result
    const fidelity  = 0.982 + Math.random() * 0.016;
    const simResult = generateMockResult(circuit.numQubits, shots, fidelity, backend);
    setResult(simResult);

    setExecuting(false);
    setPanel("results");
    addToast({
      title:   "Execution complete",
      message: `Fidelity ${fidelity.toFixed(4)} · ${shots} shots · 0.42s`,
      type:    "success",
    });
  }, [
    executing, circuit, shots, backend, maxBondDim, validation,
    setExecuting, resetSteps, setStepStatus, setResult,
    setPanel, addToast,
  ]);

  // Listen for run event from TopBar
  useEffect(() => {
    const handler = () => runCircuit();
    window.addEventListener("zuntenium:run", handler);
    return () => window.removeEventListener("zuntenium:run", handler);
  }, [runCircuit]);

  const canRun = !executing && circuit.gates.length > 0 &&
                 !validation.issues.some(i => i.severity === "error");

  return (
    <div>
      {/* Backend config */}
      <div className="mb-5">
        <div className="section-label mb-3">Backend Configuration</div>
        <div className="space-y-2">
          <ConfigRow label="Mode">
            <select
              className="config-select"
              value={optimMode}
              onChange={e => useExecutionStore.getState().setOptimMode(e.target.value as "ai"|"rule")}
            >
              <option value="ai">🤖 AI Optimizer</option>
              <option value="rule">📐 Rule-based</option>
            </select>
          </ConfigRow>
          {optimMode === "rule" && (
            <ConfigRow label="Backend">
              <select
                className="config-select"
                value={backend}
                onChange={e => useExecutionStore.getState().setBackend(e.target.value as typeof backend)}
              >
                <option value="auto">Auto</option>
                <option value="statevector">Statevector (local)</option>
                <option value="tensornetwork">MPS χ=64 (local)</option>
                <option value="ibm">IBM Quantum</option>
                <option value="ionq">IonQ Cloud</option>
                <option value="braket">AWS Braket</option>
              </select>
            </ConfigRow>
          )}
          <ConfigRow label="Shots">
            <input
              type="number" min={1} max={100_000}
              value={shots}
              onChange={e => useExecutionStore.getState().setShots(parseInt(e.target.value)||1024)}
              className="input-base flex-1 text-right font-mono"
            />
          </ConfigRow>
          <ToggleRow
            label="Error Mitigation"
            value={mitigEnabled}
            onChange={v => useExecutionStore.getState().setMitigation(v)}
          />
          <ToggleRow
            label="Return Statevector"
            value={true}
            onChange={() => {}}
          />
        </div>
      </div>

      {/* Animated pipeline */}
      <div className="mb-4">
        <div className="section-label mb-2">Execution Pipeline</div>
        <div className="space-y-[3px]">
          {steps.map((step, idx) => (
            <div key={step.id}>
              <div
                className={`flex items-center gap-2.5 px-2.5 py-[7px] rounded border
                            transition-all duration-200
                            ${step.status === "active"   ? "border-primary-DEFAULT/40 bg-primary-dim" :
                              step.status === "complete" ? "border-success/30" :
                              step.status === "error"    ? "border-danger/30"  :
                              "border-[var(--border)] bg-card"}`}
              >
                {/* Step indicator */}
                <div
                  className={`w-[7px] h-[7px] rounded-full flex-shrink-0 transition-all
                              ${step.status === "active"   ? "border-2 border-primary-DEFAULT bg-transparent animate-spin" :
                                step.status === "complete" ? "bg-success" :
                                step.status === "error"    ? "bg-danger"  :
                                "bg-[var(--border)]"}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-medium">{step.label}</div>
                  <div className="text-[9px] text-t3 font-mono truncate mt-0.5">
                    {step.detail}
                  </div>
                </div>
                {step.badge && (
                  <span className={`tag text-[8px] ${step.badgeClass ?? "tag-ok"}`}>
                    {step.badge}
                  </span>
                )}
              </div>
              {idx < steps.length - 1 && (
                <div className="w-px h-[7px] bg-[var(--border)] ml-[18px]" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={runCircuit}
        disabled={!canRun}
        className={`w-full h-9 rounded flex items-center justify-center gap-2
                    text-[12px] font-medium border transition-all duration-150
                    ${canRun
                      ? "bg-primary-DEFAULT border-primary-DEFAULT text-white hover:bg-[#006ACC]"
                      : "bg-transparent border-[var(--border)] text-t3 cursor-not-allowed"
                    }`}
      >
        {executing ? "⏳ Running…" : "▶ Run Circuit"}
      </button>

      {!canRun && !executing && circuit.gates.length === 0 && (
        <p className="text-[10px] text-t3 text-center mt-2">
          Add gates to enable execution
        </p>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

function ConfigRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-[11px] text-t2 w-[72px] flex-shrink-0">{label}</label>
      {children}
    </div>
  );
}

function ToggleRow({
  label, value, onChange,
}: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-[11px] text-t2">{label}</label>
      <button
        onClick={() => onChange(!value)}
        className={`w-[34px] h-[19px] rounded-full relative border-none cursor-pointer
                    transition-colors duration-200
                    ${value ? "bg-primary-DEFAULT" : "bg-[var(--border)]"}`}
      >
        <span
          className="absolute w-[13px] h-[13px] bg-white rounded-full top-[3px]
                     transition-all duration-200"
          style={{ left: value ? 18 : 3 }}
        />
      </button>
    </div>
  );
}

// ── Mock result generator ─────────────────────────────────────────────────
function generateMockResult(
  numQubits: number, shots: number, fidelity: number, backend: string
) {
  const dim   = 2 ** numQubits;
  const probs = Array.from({ length: dim }, () => Math.random() ** 2);
  const tot   = probs.reduce((a, b) => a + b, 0);
  const norm  = probs.map(p => p / tot);

  const counts: Record<string, number> = {};
  let remaining = shots;
  norm.forEach((p, i) => {
    const c = i === norm.length - 1 ? remaining : Math.round(p * shots);
    remaining -= c;
    if (c > 0) counts[i.toString(2).padStart(numQubits, "0")] = c;
  });

  const sv: [number, number][] = norm.map(p => {
    const amp = Math.sqrt(p);
    const phi = Math.random() * Math.PI * 2;
    return [amp * Math.cos(phi), amp * Math.sin(phi)];
  });

  return {
    job_id:             `sim-${Date.now()}`,
    status:             "completed" as const,
    counts,
    statevector:        sv,
    backend_used:       backend === "auto" ? "tensornetwork" : backend,
    decision_reason:    "tensornetwork · rl_refine · conf=0.87",
    ai_confidence:      0.87,
    fidelity_before:    fidelity - 0.04,
    fidelity_after:     fidelity,
    execution_time_s:   0.28 + Math.random() * 0.4,
    queue_time_s:       0.0,
    total_time_s:       0.35 + Math.random() * 0.5,
  };
}

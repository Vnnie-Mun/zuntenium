// src/components/layout/Footer.tsx
"use client";
import { useCircuitStore, useExecutionStore } from "@/store";

export function Footer() {
  const circuit   = useCircuitStore(s => s.circuit);
  const executing = useExecutionStore(s => s.executing);
  const result    = useExecutionStore(s => s.result);
  const gates     = circuit.gates.length;
  const depth     = circuit.gates.reduce((m, g) => Math.max(m, g.col), -1) + 1;

  const status = executing
    ? "Executing…"
    : result?.fidelity_after
    ? `F=${result.fidelity_after.toFixed(4)} · Done`
    : "Ready";

  return (
    <footer
      className="h-[23px] flex-shrink-0 flex items-center gap-3.5 px-3
                 bg-surface border-t border-[var(--border)]
                 text-[9px] font-mono text-t3"
    >
      <span>ZUNTENIUM v5.0</span>
      <span className="text-[var(--border-hi)]">|</span>
      <span>Phases 1–5 Integrated</span>
      <span className="text-[var(--border-hi)]">|</span>
      <span className="text-cyan-DEFAULT">{status}</span>
      <span className="text-[var(--border-hi)]">|</span>
      <span>{gates} gates · {circuit.numQubits} qubits · depth {depth}</span>
      <div className="flex-1" />
      <span>WS: Connected</span>
      <span className="text-[var(--border-hi)]">|</span>
      <span>API: <span className="text-cyan-DEFAULT">api.zuntenium.io</span></span>
      <span className="text-[var(--border-hi)]">|</span>
      <span>Prompts 1–6 / 18 ✓</span>
    </footer>
  );
}

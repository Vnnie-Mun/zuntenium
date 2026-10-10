// src/components/circuit/CircuitCanvas.tsx
"use client";
import {
  useCallback, useEffect, useRef, useState,
} from "react";
import { useCircuitStore, useUIStore }  from "@/store";
import { useCircuitValidation }         from "@/hooks/useCircuitValidation";
import { GATE_META }                    from "@/lib/gates";
import { GateNode }                     from "./GateNode";
import type { GateName, UIGate }        from "@/types";

const SLOT_W = 54;   // px per column
const ROW_H  = 62;   // px per qubit row

export function CircuitCanvas() {
  const circuit      = useCircuitStore(s => s.circuit);
  const selectedId   = useCircuitStore(s => s.selectedId);
  const addGate      = useCircuitStore(s => s.addGate);
  const setSelected  = useCircuitStore(s => s.setSelected);
  const setNumQubits = useCircuitStore(s => s.setNumQubits);
  const setName      = useCircuitStore(s => s.setName);
  const clearCircuit = useCircuitStore(s => s.clearCircuit);
  const undo         = useCircuitStore(s => s.undo);
  const historyIdx   = useCircuitStore(s => s.historyIdx);
  const setPanel     = useUIStore(s => s.setPanel);
  const validation   = useCircuitValidation(circuit);

  const scrollRef  = useRef<HTMLDivElement>(null);
  const [dropHint, setDropHint] = useState<{ q: number; col: number } | null>(null);

  const maxCol   = circuit.gates.reduce((m, g) => Math.max(m, g.col), -1);
  const numCols  = Math.max(14, maxCol + 3);

  // ── Keyboard shortcuts ───────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); undo(); }
      if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedId) {
          e.preventDefault();
          useCircuitStore.getState().removeGate(selectedId);
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, undo]);

  // ── Deselect on canvas click ─────────────────────────────────────────────
  function onCanvasClick(e: React.MouseEvent) {
    const tgt = e.target as HTMLElement;
    if (!tgt.closest("[data-gate-id]")) setSelected(null);
  }

  // ── Drop handler ─────────────────────────────────────────────────────────
  const onDrop = useCallback((e: React.DragEvent, q: number, col: number) => {
    e.preventDefault();
    setDropHint(null);
    const name = e.dataTransfer.getData("zuntenium/gate") as GateName;
    if (!name || !GATE_META[name]) return;
    const meta = GATE_META[name];
    const newGate: Omit<UIGate, "id"> = {
      name, qubit: q, col, params: meta.paramDefaults ?? [],
      target:  meta.isTwoQubit    ? (q < circuit.numQubits - 1 ? q + 1 : q - 1) : undefined,
      target2: meta.isThreeQubit  ? (q < circuit.numQubits - 2 ? q + 2 : q - 2) : undefined,
    };
    addGate(newGate);
  }, [addGate, circuit.numQubits]);

  // ── Validation bar state ─────────────────────────────────────────────────
  const hasErrors   = validation.issues.some(i => i.severity === "error");
  const hasWarnings = validation.issues.some(i => i.severity === "warning");
  const isEmpty     = circuit.gates.length === 0;

  const valClass = isEmpty
    ? "border-[var(--border)] text-t3"
    : hasErrors
    ? "border-danger/40 bg-danger/[0.05] text-danger"
    : hasWarnings
    ? "border-warning/40 bg-warning/[0.05] text-warning"
    : "border-success/30 bg-success/[0.05] text-success";

  const valText = isEmpty
    ? "◎ Empty circuit — add gates to run"
    : hasErrors
    ? `✕ ${validation.issues.filter(i=>i.severity==="error").length} error(s): ${validation.issues.find(i=>i.severity==="error")?.message}`
    : hasWarnings
    ? `⚠ ${validation.issues.find(i=>i.severity==="warning")?.message}`
    : `✓ Circuit valid — ${circuit.gates.length} gates, depth ${maxCol + 1}`;

  // ── Error gate IDs ───────────────────────────────────────────────────────
  const errorGateIds = new Set(
    validation.issues.filter(i => i.gateId).map(i => i.gateId!)
  );

  return (
    <main className="flex-1 relative overflow-hidden bg-base">
      {/* Grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(22,36,64,.35) 1px,transparent 1px)," +
            "linear-gradient(90deg,rgba(22,36,64,.35) 1px,transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />
      {/* Radial glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 70% 40% at 50% -10%,rgba(0,132,255,.07) 0%,transparent 60%)",
        }}
      />

      {/* Scrollable canvas */}
      <div
        ref={scrollRef}
        className="absolute inset-0 overflow-auto"
        style={{ padding: "28px 20px 20px" }}
        onClick={onCanvasClick}
      >
        {/* Circuit header */}
        <div className="flex items-center gap-3 mb-5">
          <input
            value={circuit.name}
            onChange={e => setName(e.target.value)}
            placeholder="Circuit name"
            className="bg-transparent border border-transparent rounded px-2 py-1
                       text-[17px] font-semibold text-t1 outline-none
                       hover:border-[var(--border)] focus:border-primary-DEFAULT
                       focus:bg-card transition-colors w-48"
          />
          <span className="text-[10px] text-t3 font-mono">
            n={circuit.numQubits} · d={maxCol + 1} · g={circuit.gates.length}
          </span>

          {/* Validation bar — Prompt 4 */}
          <div
            className={`flex items-center gap-2 h-[26px] px-3 rounded border
                        text-[10px] font-mono transition-all duration-200 flex-1 max-w-xs
                        ${valClass}`}
          >
            <span className="truncate">{valText}</span>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <button
              className="btn btn-sm text-[10px]"
              onClick={() => setNumQubits(circuit.numQubits + 1)}
            >
              + Qubit
            </button>
            <button
              className="btn btn-sm text-[10px]"
              onClick={undo}
              disabled={historyIdx === 0}
            >
              ↺ Undo
            </button>
            <button
              className="btn btn-sm btn-danger text-[10px]"
              onClick={clearCircuit}
            >
              ✕ Clear
            </button>
          </div>
        </div>

        {/* Qubit rows */}
        <div className="relative" style={{ minWidth: numCols * SLOT_W + 60 }}>
          {/* SVG connectors layer for 2Q gates */}
          <ConnectorsSVG circuit={circuit} numCols={numCols} />

          {Array.from({ length: circuit.numQubits }, (_, q) => (
            <QubitRow
              key={q}
              qubit={q}
              numCols={numCols}
              gates={circuit.gates.filter(g => g.qubit === q)}
              allGates={circuit.gates}
              selectedId={selectedId}
              errorGateIds={errorGateIds}
              dropHint={dropHint}
              onDrop={onDrop}
              onDropEnter={(q2, col) => setDropHint({ q: q2, col })}
              onDropLeave={() => setDropHint(null)}
              onSelect={id => {
                setSelected(id);
                setPanel("properties");
              }}
            />
          ))}

          {/* Add qubit button */}
          <div className="flex items-center mt-2" style={{ paddingLeft: 44 }}>
            <button
              onClick={() => setNumQubits(circuit.numQubits + 1)}
              className="w-9 h-9 rounded-full border border-dashed border-[var(--border)]
                         text-t3 text-base hover:border-primary-DEFAULT
                         hover:text-primary-DEFAULT hover:bg-primary-dim
                         transition-all flex items-center justify-center"
              title="Add qubit"
            >
              +
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

// ── QubitRow ──────────────────────────────────────────────────────────────
function QubitRow({
  qubit, numCols, gates, allGates, selectedId, errorGateIds,
  dropHint, onDrop, onDropEnter, onDropLeave, onSelect,
}: {
  qubit:       number;
  numCols:     number;
  gates:       UIGate[];
  allGates:    UIGate[];
  selectedId:  string | null;
  errorGateIds:Set<string>;
  dropHint:    { q: number; col: number } | null;
  onDrop:      (e: React.DragEvent, q: number, col: number) => void;
  onDropEnter: (q: number, col: number) => void;
  onDropLeave: () => void;
  onSelect:    (id: string) => void;
}) {
  const gateByCol = new Map(gates.map(g => [g.col, g]));
  // Also mark columns occupied by 2Q target qubits
  const targetCols = new Set(
    allGates
      .filter(g => g.target === qubit || g.target2 === qubit)
      .map(g => g.col)
  );

  return (
    <div className="flex items-center" style={{ height: ROW_H }}>
      {/* Qubit label */}
      <div
        className="flex-shrink-0 text-right pr-2.5 text-[11px] font-mono text-t2
                   select-none"
        style={{ width: 44 }}
      >
        Q<span className="text-primary-DEFAULT font-semibold">{qubit}</span>
      </div>

      {/* Wire + slots */}
      <div className="relative flex-1">
        {/* Horizontal wire line */}
        <div
          className="absolute left-0 right-0 pointer-events-none transition-colors duration-150"
          style={{
            top: "50%",
            height: 1.5,
            background: "var(--border-hi,#1D3058)",
            transform: "translateY(-50%)",
          }}
        />

        {/* Slots */}
        <div className="flex items-center relative z-10">
          {Array.from({ length: numCols }, (_, col) => {
            const gate    = gateByCol.get(col);
            const isHint  = dropHint?.q === qubit && dropHint?.col === col;
            const isTgt   = targetCols.has(col) && !gate;

            return (
              <div
                key={col}
                data-q={qubit}
                data-col={col}
                style={{ width: SLOT_W, height: ROW_H }}
                className={`flex items-center justify-center relative flex-shrink-0
                            transition-colors duration-100
                            ${isHint ? "bg-primary-dim" : ""}`}
                onDragOver={e => { e.preventDefault(); onDropEnter(qubit, col); }}
                onDragLeave={onDropLeave}
                onDrop={e => onDrop(e, qubit, col)}
              >
                {isHint && !gate && (
                  <div className="text-primary-DEFAULT text-lg opacity-40 pointer-events-none">
                    +
                  </div>
                )}

                {/* Target placeholder for 2Q gates */}
                {isTgt && !gate && <TargetPlaceholder qubit={qubit} col={col} allGates={allGates} />}

                {gate && (
                  <GateNode
                    gate={gate}
                    isSelected={gate.id === selectedId}
                    hasError={errorGateIds.has(gate.id)}
                    onSelect={onSelect}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── TargetPlaceholder: shows control/target icons for 2Q gates ────────────
function TargetPlaceholder({
  qubit, col, allGates,
}: {
  qubit: number; col: number; allGates: UIGate[];
}) {
  const srcGate = allGates.find(g =>
    (g.target === qubit || g.target2 === qubit) && g.col === col
  );
  if (!srcGate) return null;
  const meta  = GATE_META[srcGate.name];
  const color = meta.color;

  if (srcGate.name === "CX") {
    return (
      <svg width="22" height="22" viewBox="-11 -11 22 22" style={{ overflow: "visible" }}>
        <circle cx="0" cy="0" r="10" fill="none" stroke={color} strokeWidth="2" />
        <line x1="-8" y1="0" x2="8" y2="0" stroke={color} strokeWidth="2" />
        <line x1="0" y1="-8" x2="0" y2="8" stroke={color} strokeWidth="2" />
      </svg>
    );
  }
  return (
    <svg width="10" height="10" viewBox="-5 -5 10 10">
      <circle cx="0" cy="0" r="5" fill={color} />
    </svg>
  );
}

// ── SVG Connectors for 2Q gates ───────────────────────────────────────────
function ConnectorsSVG({ circuit, numCols }: { circuit: { numQubits: number; gates: UIGate[] }; numCols: number }) {
  const twoQGates = circuit.gates.filter(g => {
    const meta = GATE_META[g.name];
    return (meta.isTwoQubit || meta.isThreeQubit) && g.target !== undefined;
  });

  const LABEL_W = 44;
  const cx = (col: number) => LABEL_W + col * SLOT_W + SLOT_W / 2;
  const cy = (q: number)   => q * ROW_H + ROW_H / 2;

  return (
    <svg
      className="absolute inset-0 pointer-events-none overflow-visible"
      style={{ zIndex: 5 }}
      width={LABEL_W + numCols * SLOT_W}
      height={circuit.numQubits * ROW_H}
    >
      {twoQGates.map(gate => {
        const meta  = GATE_META[gate.name];
        const color = meta.color;
        const x     = cx(gate.col);
        const y1    = cy(gate.qubit);
        const y2    = cy(gate.target!);

        return (
          <g key={gate.id}>
            {/* Connector line */}
            <line
              x1={x} y1={y1} x2={x} y2={y2}
              stroke={color} strokeWidth={2}
              style={{ filter: `drop-shadow(0 0 4px ${color}80)` }}
            />
            {/* Control dot */}
            <circle
              cx={x} cy={y1} r={5} fill={color}
              style={{ filter: `drop-shadow(0 0 5px ${color})` }}
            />
          </g>
        );
      })}
    </svg>
  );
}

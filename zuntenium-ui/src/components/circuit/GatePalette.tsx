// src/components/circuit/GatePalette.tsx
"use client";
import { useState, useMemo } from "react";
import { GATE_META, GATE_CATEGORIES, CATEGORY_LABELS } from "@/lib/gates";
import type { GateName } from "@/types";
import { useCircuitStore } from "@/store";

// Color map for chip backgrounds
const CHIP_STYLE: Record<string, { bg: string; text: string; border: string }> = {
  "#0084FF": { bg:"rgba(0,132,255,.18)",   text:"#60A5FA", border:"rgba(0,132,255,.35)" },
  "#FF4757": { bg:"rgba(255,71,87,.18)",   text:"#FC8181", border:"rgba(255,71,87,.35)" },
  "#FF6B35": { bg:"rgba(255,107,53,.18)",  text:"#FBB06B", border:"rgba(255,107,53,.35)" },
  "#A855F7": { bg:"rgba(168,85,247,.18)",  text:"#C084FC", border:"rgba(168,85,247,.35)" },
  "#00D4B8": { bg:"rgba(0,212,184,.18)",   text:"#5EEAD4", border:"rgba(0,212,184,.35)" },
  "#F5A623": { bg:"rgba(245,166,35,.18)",  text:"#FCD34D", border:"rgba(245,166,35,.35)" },
  "#E91E8C": { bg:"rgba(233,30,140,.18)",  text:"#F472B6", border:"rgba(233,30,140,.35)" },
  "#7C3AED": { bg:"rgba(124,58,237,.18)",  text:"#A78BFA", border:"rgba(124,58,237,.35)" },
  "#F59E0B": { bg:"rgba(245,158,11,.18)",  text:"#FCD34D", border:"rgba(245,158,11,.35)" },
  "#10B981": { bg:"rgba(16,185,129,.18)",  text:"#6EE7B7", border:"rgba(16,185,129,.35)" },
  "#4A6080": { bg:"rgba(74,96,128,.18)",   text:"#94A3B8", border:"rgba(74,96,128,.35)" },
};

function getChipStyle(color: string) {
  return CHIP_STYLE[color] ?? { bg: "rgba(74,96,128,.18)", text: "#94A3B8", border: "rgba(74,96,128,.35)" };
}

export function GatePalette() {
  const [query, setQuery] = useState("");
  const circuit    = useCircuitStore(s => s.circuit);
  const totalGates = circuit.gates.length;
  const numQubits  = circuit.numQubits;
  const depth      = circuit.gates.reduce((m, g) => Math.max(m, g.col), -1) + 1;
  const twoQ       = circuit.gates.filter(g => GATE_META[g.name]?.isTwoQubit).length;

  const filtered = useMemo(() => {
    if (!query) return null;
    const q = query.toLowerCase();
    return Object.entries(GATE_META).filter(([name, meta]) =>
      name.toLowerCase().includes(q) ||
      meta.fullName.toLowerCase().includes(q) ||
      meta.description.toLowerCase().includes(q)
    ).map(([name]) => name as GateName);
  }, [query]);

  function onDragStart(e: React.DragEvent, name: GateName) {
    e.dataTransfer.setData("zuntenium/gate", name);
    e.dataTransfer.effectAllowed = "copy";
  }

  const SECTIONS = Object.entries(GATE_CATEGORIES) as [string, GateName[]][];

  return (
    <aside
      className="w-[232px] flex-shrink-0 flex flex-col overflow-hidden
                 bg-surface border-r border-[var(--border)]"
    >
      {/* Header + search */}
      <div className="px-3 pt-3 pb-0">
        <div className="section-label mb-2">Gate Library</div>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search gates…"
          className="w-full h-7 px-2.5 bg-card border border-[var(--border)] rounded
                     text-[11px] text-t1 placeholder-t3 outline-none
                     focus:border-primary-DEFAULT transition-colors font-sans"
        />
      </div>

      {/* Gate list */}
      <div className="flex-1 overflow-y-auto mt-2">
        {filtered ? (
          // Search results: flat list
          <div className="border-t border-[var(--border)]">
            <div className="section-label px-3 py-2">Results ({filtered.length})</div>
            {filtered.length === 0 && (
              <p className="text-[11px] text-t3 px-3 py-2">No gates found</p>
            )}
            {filtered.map(name => (
              <GateItem key={name} name={name} onDragStart={onDragStart} />
            ))}
          </div>
        ) : (
          // Categorised sections
          SECTIONS.map(([cat, names]) => (
            <div key={cat} className="border-t border-[var(--border)]">
              <div className="section-label px-3 py-2">
                {CATEGORY_LABELS[cat]}
              </div>
              {names.map(name => (
                <GateItem key={name} name={name} onDragStart={onDragStart} />
              ))}
            </div>
          ))
        )}
      </div>

      {/* Circuit stats footer */}
      <div className="border-t border-[var(--border)] p-2.5">
        <div className="section-label mb-2">Circuit Stats</div>
        <div className="grid grid-cols-2 gap-1.5">
          {[
            { label: "Qubits",   value: numQubits,  color: "text-primary-DEFAULT" },
            { label: "Gates",    value: totalGates, color: "text-cyan-DEFAULT" },
            { label: "Depth",    value: depth,      color: "text-warning" },
            { label: "2Q Gates", value: twoQ,       color: "text-purple" },
          ].map(s => (
            <div
              key={s.label}
              className="bg-card border border-[var(--border)] rounded-md
                         py-1.5 px-2 text-center"
            >
              <div className={`text-[17px] font-bold font-mono leading-none ${s.color}`}>
                {s.value}
              </div>
              <div className="text-[8px] text-t3 uppercase tracking-wide mt-1">
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function GateItem({
  name,
  onDragStart,
}: {
  name: GateName;
  onDragStart: (e: React.DragEvent, n: GateName) => void;
}) {
  const meta  = GATE_META[name];
  const chip  = getChipStyle(meta.color);

  return (
    <div
      draggable
      onDragStart={e => onDragStart(e, name)}
      className="flex items-center gap-2.5 px-3 py-[6px] cursor-grab
                 hover:bg-hover transition-colors duration-100 select-none
                 active:cursor-grabbing"
    >
      <div
        className="w-[28px] h-[25px] rounded flex items-center justify-center
                   text-[10px] font-bold font-mono flex-shrink-0 tracking-tight"
        style={{
          background: chip.bg,
          color:      chip.text,
          border:     `1px solid ${chip.border}`,
        }}
      >
        {meta.label}
      </div>
      <div className="min-w-0">
        <div className="text-[11px] font-medium leading-[1.2] truncate">
          {meta.fullName}
        </div>
        <div className="text-[9px] text-t3 mt-0.5 truncate">
          {meta.description}
        </div>
      </div>
    </div>
  );
}

// src/components/circuit/ParameterPanel.tsx
"use client";
import { GATE_META, formatAngle, parseAngle, ANGLE_PRESETS } from "@/lib/gates";
import { useCircuitStore } from "@/store";

export function ParameterPanel({ gateId }: { gateId: string }) {
  const gate       = useCircuitStore(s => s.circuit.gates.find(g => g.id === gateId));
  const numQubits  = useCircuitStore(s => s.circuit.numQubits);
  const updateGate = useCircuitStore(s => s.updateGate);
  const removeGate = useCircuitStore(s => s.removeGate);
  const addGate    = useCircuitStore(s => s.addGate);

  if (!gate) {
    return (
      <div className="flex flex-col items-center py-8 text-t3 text-[11px] text-center px-4">
        <div className="text-2xl mb-2 opacity-40">⊡</div>
        <div className="text-t2 text-xs mb-1">No gate selected</div>
        <div>Click a gate on the canvas to<br />inspect and edit its properties</div>
      </div>
    );
  }

  const meta  = GATE_META[gate.name];
  const color = meta.color;
  const theta = gate.params[0] ?? (meta.paramDefaults?.[0] ?? 0);

  function setParam(v: number) {
    if (!gate) return;
    updateGate(gate.id, { params: [v] });
  }

  function handleParamInput(raw: string) {
    const v = parseAngle(raw);
    if (v !== null) setParam(v);
  }

  function handleDuplicate() {
    if (!gate) return;
    addGate({
      name: gate.name,
      qubit: gate.qubit,
      target: gate.target,
      target2: gate.target2,
      params: [...gate.params],
      col: gate.col + 1,
    });
  }

  return (
    <div>
      {/* Gate header */}
      <div className="flex items-center gap-2.5 mb-4">
        <div
          className="w-[42px] h-[42px] rounded-lg flex items-center justify-center
                     font-mono text-[13px] font-bold flex-shrink-0"
          style={{
            background: meta.bg,
            border:     `1px solid ${color}50`,
            color,
          }}
        >
          {meta.label}
        </div>
        <div>
          <div className="text-[14px] font-semibold">{gate.name} Gate</div>
          <div className="text-[10px] text-t3 mt-0.5 leading-snug">{meta.description}</div>
        </div>
      </div>

      {/* Placement */}
      <div className="mb-4">
        <div className="section-label mb-2">Placement</div>
        <div className="flex items-center gap-2 mb-2">
          <label className="text-[11px] text-t2 w-[54px] flex-shrink-0">Qubit</label>
          <select
            value={gate.qubit}
            onChange={e => updateGate(gate.id, { qubit: parseInt(e.target.value) })}
            className="flex-1 h-7 px-2 bg-card border border-[var(--border)] rounded
                       text-[11px] text-t1 outline-none focus:border-primary-DEFAULT
                       cursor-pointer"
          >
            {Array.from({ length: numQubits }, (_, i) => (
              <option key={i} value={i}>Q{i}</option>
            ))}
          </select>
        </div>

        {(meta.isTwoQubit || meta.isThreeQubit) && (
          <div className="flex items-center gap-2">
            <label className="text-[11px] text-t2 w-[54px] flex-shrink-0">Target</label>
            <select
              value={gate.target ?? ""}
              onChange={e => updateGate(gate.id, { target: parseInt(e.target.value) })}
              className="flex-1 h-7 px-2 bg-card border border-[var(--border)] rounded
                         text-[11px] text-t1 outline-none focus:border-primary-DEFAULT
                         cursor-pointer"
            >
              {Array.from({ length: numQubits }, (_, i) => (
                <option key={i} value={i}>Q{i}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Prompt 5: Full parameter editing block */}
      {meta.hasParams && (
        <div
          className="bg-card border border-[var(--border)] rounded-lg p-3 mb-4"
        >
          <div className="section-label mb-3">Parameters</div>

          {/* Slider */}
          <div className="flex items-center gap-2 mb-3">
            <span
              className="text-[11px] font-mono flex-shrink-0 w-4 text-t2"
            >
              {meta.paramNames?.[0] ?? "θ"}
            </span>
            <input
              type="range"
              min={-Math.PI * 2}
              max={Math.PI * 2}
              step={0.01}
              value={theta}
              onChange={e => setParam(parseFloat(e.target.value))}
              className="flex-1 cursor-pointer"
              style={{ accentColor: color }}
            />
            <span
              className="text-[11px] font-mono text-right w-[44px] flex-shrink-0"
              style={{ color: "var(--cyan,#00D4B8)" }}
            >
              {formatAngle(theta)}
            </span>
          </div>

          {/* Numeric input */}
          <div className="flex items-center gap-2 mb-3">
            <label className="text-[11px] text-t2 w-[54px] flex-shrink-0">Value</label>
            <input
              className="flex-1 h-7 px-2 bg-surface border border-[var(--border)] rounded
                         text-[11px] font-mono text-t1 outline-none
                         focus:border-primary-DEFAULT"
              defaultValue={formatAngle(theta)}
              key={formatAngle(theta)}   // re-mount when value changes externally
              onBlur={e => handleParamInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") handleParamInput((e.target as HTMLInputElement).value); }}
              placeholder="e.g. π/2, 1.5708"
            />
          </div>

          {/* Angle presets */}
          <div>
            <div className="text-[9px] text-t3 uppercase tracking-wide mb-1.5">Presets</div>
            <div className="flex flex-wrap gap-1">
              {ANGLE_PRESETS.map(preset => {
                const isActive = Math.abs(theta - preset.value) < 0.01;
                return (
                  <button
                    key={preset.label}
                    onClick={() => setParam(preset.value)}
                    className={`h-[22px] px-2 rounded border text-[9px] font-mono
                                transition-all duration-100
                                ${isActive
                                  ? "border-cyan-DEFAULT text-cyan-DEFAULT bg-cyan-dim"
                                  : "border-[var(--border)] text-t2 bg-transparent hover:border-primary-DEFAULT hover:text-primary-DEFAULT hover:bg-primary-dim"
                                }`}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Live preview badge */}
          <div
            className="mt-3 flex items-center justify-between px-2 py-1.5
                       rounded bg-surface border border-[var(--border)]"
          >
            <span className="text-[10px] text-t3 font-mono">Live label</span>
            <span
              className="text-[11px] font-mono font-bold"
              style={{ color }}
            >
              {meta.label}({formatAngle(theta)})
            </span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div>
        <div className="section-label mb-2">Actions</div>
        <div className="flex gap-1.5">
          <button className="btn btn-sm flex-1 text-[10px]" onClick={handleDuplicate}>
            ⧉ Duplicate
          </button>
          <button
            className="btn btn-sm flex-1 text-[10px]"
            style={{ borderColor: "rgba(255,71,87,.4)", color: "var(--danger,#FF4757)" }}
            onClick={() => removeGate(gate.id)}
          >
            ✕ Delete
          </button>
        </div>
      </div>
    </div>
  );
}

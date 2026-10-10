// src/components/visualization/StatevectorChart.tsx
"use client";
import { useMemo, useState } from "react";

interface Props {
  statevector: [number, number][];   // [[real, imag], ...]
  numQubits:   number;
}

export function StatevectorChart({ statevector, numQubits }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

  const entries = useMemo(() => {
    const max = Math.max(
      ...statevector.map(([r, i]) => Math.sqrt(r * r + i * i)),
      1e-10
    );
    return statevector.slice(0, Math.min(2 ** numQubits, 16)).map(([re, im], idx) => {
      const mag = Math.sqrt(re * re + im * im);
      const bs  = idx.toString(2).padStart(numQubits, "0");
      return {
        idx, bs,
        re, im, mag,
        reNorm: re  / max,
        imNorm: im  / max,
        magPct: (mag / max) * 100,
      };
    });
  }, [statevector, numQubits]);

  return (
    <div>
      {/* Legend */}
      <div className="flex gap-4 mb-3">
        {[
          { color: "rgba(0,132,255,.85)", label: "Real" },
          { color: "rgba(0,212,184,.85)", label: "Imaginary" },
        ].map(l => (
          <div key={l.label} className="flex items-center gap-1.5 text-[9px] text-t3">
            <div className="w-2 h-2 rounded-sm" style={{ background: l.color }} />
            {l.label}
          </div>
        ))}
        <span className="text-[9px] text-t3 ml-auto">
          {entries.length} of {2 ** numQubits} states
        </span>
      </div>

      {/* Bars */}
      <div className="space-y-[3px]">
        {entries.map(e => (
          <div
            key={e.idx}
            className="flex items-center gap-1.5 cursor-default"
            onMouseEnter={() => setHovered(e.idx)}
            onMouseLeave={() => setHovered(null)}
          >
            {/* Basis state label */}
            <span
              className="font-mono text-[8px] flex-shrink-0 text-right"
              style={{ width: 32, color: hovered === e.idx ? "var(--t1)" : "var(--t3)" }}
            >
              |{e.bs}⟩
            </span>

            {/* Amplitude bars (bidirectional) */}
            <div className="flex-1 relative" style={{ height: 22 }}>
              {/* Center line */}
              <div
                className="absolute inset-y-0 pointer-events-none"
                style={{ left: "50%", width: 1, background: "rgba(255,255,255,.07)" }}
              />

              {/* Real bar */}
              <div
                className="absolute rounded-[2px] transition-all duration-300"
                style={{
                  height: 9, top: 2,
                  background: "rgba(0,132,255,.85)",
                  width: `${Math.abs(e.reNorm) * 46}%`,
                  [e.re >= 0 ? "left" : "right"]: "50%",
                }}
              />

              {/* Imaginary bar */}
              <div
                className="absolute rounded-[2px] transition-all duration-300"
                style={{
                  height: 9, bottom: 2,
                  background: "rgba(0,212,184,.85)",
                  width: `${Math.abs(e.imNorm) * 46}%`,
                  [e.im >= 0 ? "left" : "right"]: "50%",
                }}
              />
            </div>

            {/* Magnitude */}
            <span
              className="font-mono text-[8px] flex-shrink-0 text-right"
              style={{ width: 26, color: "var(--t3)" }}
            >
              {(e.mag * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      {/* Tooltip */}
      {hovered !== null && entries[hovered] && (
        <div
          className="mt-3 p-2 bg-surface border border-[var(--border)] rounded
                     text-[10px] font-mono text-t2"
        >
          |{entries[hovered]!.bs}⟩ &nbsp;
          re={entries[hovered]!.re.toFixed(4)} &nbsp;
          im={entries[hovered]!.im.toFixed(4)} &nbsp;
          |ψ|={entries[hovered]!.mag.toFixed(4)} &nbsp;
          P={(entries[hovered]!.mag ** 2).toFixed(4)}
        </div>
      )}
    </div>
  );
}

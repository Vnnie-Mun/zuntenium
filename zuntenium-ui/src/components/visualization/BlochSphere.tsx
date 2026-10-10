// src/components/visualization/BlochSphere.tsx
"use client";
import { useMemo } from "react";

interface Props {
  /** probability of |0⟩ for this qubit */
  p0:        number;
  qubitIdx?: number;
}

export function BlochSphere({ p0, qubitIdx = 0 }: Props) {
  const R = 54;

  const { vx, vy } = useMemo(() => {
    // Map p0 to Bloch polar angle θ: p0 = cos²(θ/2) ⟹ θ = 2·arccos(√p0)
    const clamped = Math.max(0, Math.min(1, p0));
    const theta   = 2 * Math.acos(Math.sqrt(clamped));
    // Use a fixed phi for display (deterministic from p0 for reproducibility)
    const phi = (p0 * 2.718) % (Math.PI * 2);
    return {
      vx: R * Math.sin(theta) * Math.cos(phi),
      vy: -R * Math.cos(theta),
    };
  }, [p0]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-[9px] text-t3 uppercase tracking-[.8px] mb-2">
        Bloch Sphere — Q<span className="text-primary-DEFAULT font-semibold">{qubitIdx}</span>
      </div>
      <svg
        width={140} height={140}
        viewBox="-65 -65 130 130"
        style={{ overflow: "visible" }}
      >
        {/* Sphere outline */}
        <circle cx={0} cy={0} r={R} fill="none" stroke="#162440" strokeWidth={1} />

        {/* Meridians */}
        <ellipse
          cx={0} cy={0} rx={R} ry={R * 0.32}
          fill="none" stroke="#162440" strokeWidth={1} strokeDasharray="3 3"
        />
        <ellipse
          cx={0} cy={0} rx={R * 0.32} ry={R}
          fill="none" stroke="#162440" strokeWidth={1} strokeDasharray="3 3" opacity={0.5}
        />

        {/* Axes */}
        <line x1={0}  y1={-62} x2={0}  y2={62}  stroke="#1D3058" strokeWidth={1} />
        <line x1={-62} y1={0}  x2={62} y2={0}   stroke="#1D3058" strokeWidth={1} />

        {/* Axis labels */}
        {[
          [0, -70, "+Z"], [0, 76, "-Z"], [68, 3, "X"], [-68, 3, ""],
        ].map(([x, y, label]) => (
          <text
            key={String(label)}
            x={x} y={y}
            textAnchor="middle"
            fill="#2A3F60"
            fontSize={9}
            fontFamily="IBM Plex Mono, monospace"
          >
            {label}
          </text>
        ))}

        {/* State vector */}
        <line
          x1={0} y1={0}
          x2={vx} y2={vy}
          stroke="#00D4B8" strokeWidth={2.5} strokeLinecap="round"
          style={{ filter: "drop-shadow(0 0 4px #00D4B8)" }}
        />

        {/* State vector tip */}
        <circle
          cx={vx} cy={vy} r={5}
          fill="#00D4B8"
          style={{ filter: "drop-shadow(0 0 6px #00D4B8)" }}
        />

        {/* Origin dot */}
        <circle cx={0} cy={0} r={2} fill="#2A3F60" />
      </svg>

      {/* Coordinates display */}
      <div className="mt-1 text-[9px] font-mono text-t3 text-center">
        P(|0⟩) = {p0.toFixed(4)} &nbsp; P(|1⟩) = {(1 - p0).toFixed(4)}
      </div>
    </div>
  );
}

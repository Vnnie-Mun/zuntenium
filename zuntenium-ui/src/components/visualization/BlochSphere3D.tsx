// src/components/visualization/BlochSphere3D.tsx
// Prompt 7 — Interactive 3D Bloch sphere (pure SVG/canvas — no Three.js dependency needed)
// Uses perspective projection + rotation to give a full 3D feel.
"use client";
import { useRef, useState, useEffect, useCallback, memo } from "react";

interface BlochState {
  theta: number;   // polar angle 0..π (0=|0⟩, π=|1⟩)
  phi:   number;   // azimuthal angle 0..2π
}

interface Props {
  /** Probability of |0⟩ state */
  p0:        number;
  /** Imaginary component for phase (optional) */
  phase?:    number;
  qubitIdx?: number;
  size?:     number;
  animate?:  boolean;
}

// 3D → 2D perspective projection
function project(x: number, y: number, z: number, rotX: number, rotY: number, dist = 280) {
  // Rotate around Y axis
  const x1 = x * Math.cos(rotY) + z * Math.sin(rotY);
  const z1 = -x * Math.sin(rotY) + z * Math.cos(rotY);
  // Rotate around X axis
  const y2 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
  const z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);
  const scale = dist / (dist + z2 + 80);
  return { px: x1 * scale, py: y2 * scale, z: z2, scale };
}

function toBloch(p0: number, phase: number): BlochState {
  const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p0));
  return {
    theta: 2 * Math.acos(Math.sqrt(clamped)),
    phi:   phase,
  };
}

export const BlochSphere3D = memo(function BlochSphere3D({
  p0,
  phase = 0,
  qubitIdx = 0,
  size    = 200,
  animate = true,
}: Props) {
  const R   = size * 0.38;
  const cx  = size / 2;
  const cy  = size / 2;

  const [rotX, setRotX] = useState(-0.4);
  const [rotY, setRotY] = useState(0.6);
  const [dragging, setDragging] = useState(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);
  const autoRotRef = useRef<number | null>(null);

  const bloch = toBloch(p0, phase);

  // State vector tip in Bloch coordinates
  const vx3 = R * Math.sin(bloch.theta) * Math.cos(bloch.phi);
  const vy3 = -R * Math.cos(bloch.theta);
  const vz3 = R * Math.sin(bloch.theta) * Math.sin(bloch.phi);

  // Auto-rotate when idle
  useEffect(() => {
    if (!animate || dragging) return;
    const id = setInterval(() => {
      setRotY(r => r + 0.008);
    }, 50);
    autoRotRef.current = id as unknown as number;
    return () => clearInterval(id);
  }, [animate, dragging]);

  // Drag to rotate
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    setDragging(true);
    if (autoRotRef.current) clearInterval(autoRotRef.current);
    lastPos.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging || !lastPos.current) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    setRotY(r => r + dx * 0.012);
    setRotX(r => r + dy * 0.012);
    lastPos.current = { x: e.clientX, y: e.clientY };
  }, [dragging]);

  const onMouseUp = useCallback(() => setDragging(false), []);

  function reset() {
    setRotX(-0.4);
    setRotY(0.6);
  }

  // Build sphere wireframe — latitudes and longitudes
  const LATS = 8, LONS = 8;
  const wireColor = "rgba(22,36,64,0.9)";

  function latCircle(lat: number): string {
    const y = R * Math.cos(lat);
    const r = R * Math.sin(lat);
    const pts: string[] = [];
    for (let a = 0; a <= 64; a++) {
      const phi = (a / 64) * Math.PI * 2;
      const p   = project(r * Math.cos(phi), y, r * Math.sin(phi), rotX, rotY);
      pts.push(`${cx + p.px},${cy + p.py}`);
    }
    return "M " + pts.join(" L ");
  }

  function lonCircle(lon: number): string {
    const pts: string[] = [];
    for (let a = 0; a <= 32; a++) {
      const lat = (a / 32) * Math.PI;
      const p   = project(
        R * Math.sin(lat) * Math.cos(lon),
        -R * Math.cos(lat),
        R * Math.sin(lat) * Math.sin(lon),
        rotX, rotY
      );
      pts.push(`${cx + p.px},${cy + p.py}`);
    }
    return "M " + pts.join(" L ");
  }

  // Axes
  function axisLine(ax: number, ay: number, az: number, scale = 1.2) {
    const a = project(ax * R * scale, ay * R * scale, az * R * scale, rotX, rotY);
    const b = project(-ax * R * 0.2, -ay * R * 0.2, -az * R * 0.2, rotX, rotY);
    return { ax: cx + a.px, ay: cy + a.py, bx: cx + b.px, by: cy + b.py, za: a.z };
  }

  const axes = [
    { label: "+Z", ...axisLine(0, 1, 0), color: "#4A6080" },
    { label: "-Z", ...axisLine(0, -1, 0), color: "#2A3F60" },
    { label: "X",  ...axisLine(1, 0, 0), color: "#4A6080" },
    { label: "Y",  ...axisLine(0, 0, 1), color: "#4A6080" },
  ];

  // State vector projection
  const tip    = project(vx3, vy3, vz3, rotX, rotY);
  const origin = project(0, 0, 0, rotX, rotY);

  // Trace trail on sphere (ghost path for prev positions)
  const trailPts = Array.from({ length: 6 }, (_, i) => {
    const t = (i + 1) / 6;
    const trailPhi = bloch.phi - 0.4 * t;
    const tv = project(
      R * Math.sin(bloch.theta) * Math.cos(trailPhi),
      -R * Math.cos(bloch.theta),
      R * Math.sin(bloch.theta) * Math.sin(trailPhi),
      rotX, rotY
    );
    return { x: cx + tv.px, y: cy + tv.py, a: 1 - t };
  }).reverse();

  return (
    <div className="flex flex-col items-center select-none">
      <div
        className="text-[9px] text-t3 uppercase tracking-[.8px] mb-1.5 flex items-center gap-2"
      >
        Bloch Sphere — Q<span className="text-primary-DEFAULT font-semibold">{qubitIdx}</span>
        <button
          onClick={reset}
          className="text-[8px] text-t3 hover:text-primary-DEFAULT transition-colors
                     border border-[var(--border)] px-1.5 py-0.5 rounded"
        >
          Reset
        </button>
      </div>

      <svg
        width={size} height={size}
        style={{ cursor: dragging ? "grabbing" : "grab" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {/* Sphere body — back hemisphere */}
        <circle
          cx={cx} cy={cy} r={R}
          fill="none"
          stroke="rgba(22,36,64,0.6)"
          strokeWidth={1}
        />

        {/* Wireframe latitudes */}
        {Array.from({ length: LATS - 1 }, (_, i) => {
          const lat = ((i + 1) / LATS) * Math.PI;
          return (
            <path
              key={`lat-${i}`}
              d={latCircle(lat)}
              fill="none"
              stroke={wireColor}
              strokeWidth={0.5}
              opacity={0.6}
            />
          );
        })}

        {/* Wireframe longitudes */}
        {Array.from({ length: LONS }, (_, i) => {
          const lon = (i / LONS) * Math.PI;
          return (
            <path
              key={`lon-${i}`}
              d={lonCircle(lon)}
              fill="none"
              stroke={wireColor}
              strokeWidth={0.5}
              opacity={0.6}
            />
          );
        })}

        {/* Axes */}
        {axes.map(a => (
          <g key={a.label}>
            <line
              x1={a.bx} y1={a.by}
              x2={a.ax} y2={a.ay}
              stroke="#1D3058"
              strokeWidth={1}
            />
            <text
              x={a.ax + (a.ax - cx) * 0.12}
              y={a.ay + (a.ay - cy) * 0.12 + 3}
              textAnchor="middle"
              fill={a.color}
              fontSize={8}
              fontFamily="IBM Plex Mono, monospace"
            >
              {a.label}
            </text>
          </g>
        ))}

        {/* Trail */}
        {trailPts.map((pt, i) => (
          <circle
            key={i}
            cx={pt.x} cy={pt.y}
            r={2.5 - i * 0.3}
            fill="#00D4B8"
            opacity={pt.a * 0.3}
          />
        ))}

        {/* State vector shaft */}
        <line
          x1={cx + origin.px}
          y1={cy + origin.py}
          x2={cx + tip.px}
          y2={cy + tip.py}
          stroke="#0084FF"
          strokeWidth={2.5}
          strokeLinecap="round"
          style={{ filter: "drop-shadow(0 0 5px rgba(0,132,255,0.7))" }}
        />

        {/* Arrow head */}
        <circle
          cx={cx + tip.px}
          cy={cy + tip.py}
          r={6}
          fill="#00D4B8"
          style={{ filter: "drop-shadow(0 0 8px rgba(0,212,184,0.8))" }}
        />
        <circle
          cx={cx + tip.px}
          cy={cy + tip.py}
          r={3.5}
          fill="#fff"
          opacity={0.9}
        />

        {/* Origin dot */}
        <circle cx={cx} cy={cy} r={2} fill="#2A3F60" />
      </svg>

      {/* State info */}
      <div className="mt-2 text-center font-mono">
        <div className="text-[9px] text-t3">
          θ = {bloch.theta.toFixed(3)} rad &nbsp; φ = {bloch.phi.toFixed(3)} rad
        </div>
        <div className="text-[9px] text-t3 mt-0.5">
          P(|0⟩) = {p0.toFixed(4)} &nbsp; P(|1⟩) = {(1 - p0).toFixed(4)}
        </div>
        <div className="text-[8px] text-t4 mt-0.5">
          Drag to rotate · auto-spins when idle
        </div>
      </div>
    </div>
  );
});

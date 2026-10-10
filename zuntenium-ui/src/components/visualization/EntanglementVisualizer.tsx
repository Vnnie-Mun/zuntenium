// src/components/visualization/EntanglementVisualizer.tsx
// Prompt 7 — Multi-qubit entanglement heatmap + chord diagram
"use client";
import { useState, memo } from "react";

interface Props {
  /** Pairwise entanglement matrix [n×n] where entry [i][j] ∈ [0,1] */
  matrix:    number[][];
  numQubits: number;
}

type ViewMode = "heatmap" | "chord";

function binaryEntropy(probability: number): number {
  return probability > 0 ? -probability * Math.log2(probability) : 0;
}

// Compute a mock entanglement matrix from statevector probs
export function computeEntanglementMatrix(
  statevector: [number, number][],
  numQubits:   number
): number[][] {
  const n   = numQubits;
  const dim = 2 ** n;
  const probs = statevector.map(([r, i]) => r * r + i * i);

  // Reduced density matrix marginals for pairs
  const mat = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let qi = 0; qi < n; qi++) {
    for (let qj = qi + 1; qj < n; qj++) {
      // Concurrence-like estimate via marginal entropy
      let pij00 = 0, pij01 = 0, pij10 = 0, pij11 = 0;
      for (let state = 0; state < dim; state++) {
        const bi = (state >> (n - 1 - qi)) & 1;
        const bj = (state >> (n - 1 - qj)) & 1;
        const p  = probs[state] ?? 0;
        if      (bi === 0 && bj === 0) pij00 += p;
        else if (bi === 0 && bj === 1) pij01 += p;
        else if (bi === 1 && bj === 0) pij10 += p;
        else                           pij11 += p;
      }
      // Mutual information proxy
      const px0 = pij00 + pij01, px1 = pij10 + pij11;
      const py0 = pij00 + pij10, py1 = pij01 + pij11;
      const HX  = binaryEntropy(px0) + binaryEntropy(px1);
      const HY  = binaryEntropy(py0) + binaryEntropy(py1);
      const HXY = binaryEntropy(pij00) + binaryEntropy(pij01)
        + binaryEntropy(pij10) + binaryEntropy(pij11);
      const MI  = Math.max(0, (HX + HY - HXY) / Math.log2(2));   // normalise to [0,1]
      const val = Math.min(1, MI);
      mat[qi]![qj] = val;
      mat[qj]![qi] = val;
    }
  }
  return mat;
}

export const EntanglementVisualizer = memo(function EntanglementVisualizer({
  matrix, numQubits,
}: Props) {
  const [mode, setMode]   = useState<ViewMode>("heatmap");
  const [hovered, setHov] = useState<[number, number] | null>(null);

  // Color interpolation: 0 = very dark, 1 = bright cyan
  function cellColor(v: number) {
    // 0→bg, 0.5→blue, 1→cyan
    if (v < 0.001) return "rgba(22,36,64,0.5)";
    const r = Math.round(v * 0 + (1 - v) * 0);
    const g = Math.round(v * 212);
    const b = Math.round(v * 184 + (1 - v) * 255);
    return `rgba(${r},${g},${b},${0.2 + v * 0.8})`;
  }

  return (
    <div>
      {/* Toggle */}
      <div className="flex gap-1 mb-3">
        {(["heatmap", "chord"] as ViewMode[]).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex-1 h-7 rounded border text-[10px] font-semibold
                        capitalize transition-all duration-100
                        ${mode === m
                          ? "border-primary-DEFAULT/40 bg-primary-dim text-primary-DEFAULT"
                          : "border-[var(--border)] text-t2 hover:bg-hover"
                        }`}
          >
            {m}
          </button>
        ))}
      </div>

      {mode === "heatmap" ? (
        <HeatMap
          matrix={matrix} numQubits={numQubits}
          hovered={hovered} setHovered={setHov}
          cellColor={cellColor}
        />
      ) : (
        <ChordDiagram matrix={matrix} numQubits={numQubits} />
      )}

      {/* Tooltip */}
      {hovered && (
        <div className="mt-2 px-2 py-1.5 bg-surface border border-[var(--border)] rounded
                        text-[10px] font-mono text-t2">
          Q{hovered[0]} ↔ Q{hovered[1]}: MI = {(matrix[hovered[0]]?.[hovered[1]] ?? 0).toFixed(4)}
          &nbsp;
          {(matrix[hovered[0]]?.[hovered[1]] ?? 0) > 0.5
            ? "🔗 Strongly entangled"
            : (matrix[hovered[0]]?.[hovered[1]] ?? 0) > 0.1
            ? "≈ Partially entangled"
            : "○ Separable"
          }
        </div>
      )}

      {/* Color scale */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-[9px] text-t3">0</span>
        <div
          className="flex-1 h-[5px] rounded-full"
          style={{
            background: "linear-gradient(90deg, rgba(22,36,64,0.6), #0084FF, #00D4B8)",
          }}
        />
        <span className="text-[9px] text-t3">1</span>
        <span className="text-[9px] text-t3 ml-1">MI</span>
      </div>
    </div>
  );
});

// ── Heatmap ────────────────────────────────────────────────────────────────
function HeatMap({
  matrix, numQubits, hovered, setHovered, cellColor,
}: {
  matrix:     number[][];
  numQubits:  number;
  hovered:    [number, number] | null;
  setHovered: (v: [number, number] | null) => void;
  cellColor:  (v: number) => string;
}) {
  const CELL = Math.min(Math.floor(240 / (numQubits + 1)), 32);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        width={(numQubits + 1) * CELL}
        height={(numQubits + 1) * CELL}
        style={{ display: "block" }}
      >
        {/* Column labels */}
        {Array.from({ length: numQubits }, (_, j) => (
          <text
            key={`cl-${j}`}
            x={CELL + j * CELL + CELL / 2}
            y={CELL - 3}
            textAnchor="middle"
            fill="var(--t3)"
            fontSize={9}
            fontFamily="IBM Plex Mono, monospace"
          >
            Q{j}
          </text>
        ))}

        {/* Row labels */}
        {Array.from({ length: numQubits }, (_, i) => (
          <text
            key={`rl-${i}`}
            x={CELL - 4}
            y={CELL + i * CELL + CELL / 2 + 3}
            textAnchor="end"
            fill="var(--t3)"
            fontSize={9}
            fontFamily="IBM Plex Mono, monospace"
          >
            Q{i}
          </text>
        ))}

        {/* Cells */}
        {Array.from({ length: numQubits }, (_, i) =>
          Array.from({ length: numQubits }, (_, j) => {
            const val  = i === j ? 1 : (matrix[i]?.[j] ?? 0);
            const isHov = hovered?.[0] === i && hovered?.[1] === j;
            const color = i === j ? "rgba(0,132,255,0.3)" : cellColor(val);
            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={CELL + j * CELL} y={CELL + i * CELL}
                  width={CELL - 2} height={CELL - 2}
                  rx={2}
                  fill={color}
                  stroke={isHov ? "var(--cyan)" : "transparent"}
                  strokeWidth={isHov ? 1.5 : 0}
                  style={{ cursor: "crosshair", transition: "all .15s" }}
                  onMouseEnter={() => i !== j && setHovered([i, j])}
                  onMouseLeave={() => setHovered(null)}
                />
                {i !== j && val > 0.05 && (
                  <text
                    x={CELL + j * CELL + CELL / 2 - 1}
                    y={CELL + i * CELL + CELL / 2 + 3}
                    textAnchor="middle"
                    fill="rgba(255,255,255,0.7)"
                    fontSize={7}
                    fontFamily="IBM Plex Mono, monospace"
                    style={{ pointerEvents: "none" }}
                  >
                    {val.toFixed(2)}
                  </text>
                )}
                {i === j && (
                  <text
                    x={CELL + j * CELL + CELL / 2 - 1}
                    y={CELL + i * CELL + CELL / 2 + 3}
                    textAnchor="middle"
                    fill="rgba(0,132,255,0.8)"
                    fontSize={7}
                    fontFamily="IBM Plex Mono, monospace"
                    style={{ pointerEvents: "none" }}
                  >
                    Q{i}
                  </text>
                )}
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
}

// ── Chord diagram ─────────────────────────────────────────────────────────
function ChordDiagram({ matrix, numQubits }: { matrix: number[][]; numQubits: number }) {
  const SIZE = 200;
  const cx   = SIZE / 2, cy = SIZE / 2, R = 72;

  const nodes = Array.from({ length: numQubits }, (_, i) => {
    const a = (i / numQubits) * Math.PI * 2 - Math.PI / 2;
    return { i, x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a };
  });

  function chordPath(i: number, j: number) {
    const ni = nodes[i]!, nj = nodes[j]!;
    const mx = cx, my = cy;   // control through center
    return `M${ni.x},${ni.y} Q${mx},${my} ${nj.x},${nj.y}`;
  }

  const CYAN = "var(--cyan,#00D4B8)";
  const BLUE = "var(--blue,#0084FF)";

  return (
    <svg width={SIZE} height={SIZE} style={{ display: "block", margin: "0 auto" }}>
      {/* Chord arcs */}
      {nodes.map(ni =>
        nodes
          .filter(nj => nj.i > ni.i)
          .map(nj => {
            const val = matrix[ni.i]?.[nj.i] ?? 0;
            if (val < 0.02) return null;
            return (
              <path
                key={`${ni.i}-${nj.i}`}
                d={chordPath(ni.i, nj.i)}
                fill="none"
                stroke={val > 0.5 ? CYAN : BLUE}
                strokeWidth={Math.max(0.5, val * 5)}
                opacity={0.2 + val * 0.7}
                style={{ filter: val > 0.6 ? `drop-shadow(0 0 4px ${CYAN})` : "none" }}
              />
            );
          })
      )}

      {/* Node circles */}
      {nodes.map(n => (
        <g key={n.i}>
          <circle
            cx={n.x} cy={n.y} r={8}
            fill="var(--card,#0B1728)"
            stroke={BLUE} strokeWidth={1.5}
          />
          <text
            x={n.x} y={n.y + 3.5}
            textAnchor="middle"
            fill={BLUE}
            fontSize={8}
            fontFamily="IBM Plex Mono, monospace"
          >
            Q{n.i}
          </text>
        </g>
      ))}
    </svg>
  );
}

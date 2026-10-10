// src/components/visualization/ProbabilityHistogram.tsx
"use client";
import { useMemo, useState } from "react";

interface Props {
  counts:   Record<string, number>;
  shots:    number;
  maxBars?: number;
}

export function ProbabilityHistogram({ counts, shots, maxBars = 12 }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const sorted = useMemo(() => {
    const entries = Object.entries(counts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, maxBars);
    const maxC = Math.max(...entries.map(([, c]) => c), 1);
    return entries.map(([bs, cnt]) => ({
      bs, cnt,
      pct:     (cnt / shots * 100),
      barPct:  (cnt / maxC * 100),
    }));
  }, [counts, shots, maxBars]);

  if (sorted.length === 0) return (
    <div className="text-[11px] text-t3 text-center py-4">No measurement data</div>
  );

  return (
    <div className="space-y-[5px]">
      {sorted.map(row => (
        <div
          key={row.bs}
          className="flex items-center gap-2 cursor-default group"
          onMouseEnter={() => setHovered(row.bs)}
          onMouseLeave={() => setHovered(null)}
        >
          {/* Basis state */}
          <span
            className="font-mono text-[9px] flex-shrink-0 text-right transition-colors"
            style={{
              width:  34,
              color:  hovered === row.bs ? "var(--t1)" : "var(--t2)",
            }}
          >
            |{row.bs}⟩
          </span>

          {/* Bar */}
          <div
            className="flex-1 relative rounded-[2px] overflow-hidden"
            style={{ height: 16, background: "var(--border,#162440)" }}
          >
            <div
              className="h-full rounded-[2px] transition-all duration-500"
              style={{
                width: `${row.barPct}%`,
                background: "linear-gradient(90deg,#0084FF,#00D4B8)",
                boxShadow: hovered === row.bs ? "0 0 8px rgba(0,132,255,.5)" : "none",
              }}
            />
            {/* Count label inside bar for wide bars */}
            {row.barPct > 30 && (
              <span
                className="absolute right-1.5 inset-y-0 flex items-center
                           font-mono text-[8px] text-white/80"
              >
                {row.cnt}
              </span>
            )}
          </div>

          {/* Percentage */}
          <span
            className="font-mono text-[9px] text-right flex-shrink-0"
            style={{ width: 36, color: "var(--t3)" }}
          >
            {row.pct.toFixed(1)}%
          </span>
        </div>
      ))}

      {Object.keys(counts).length > maxBars && (
        <div className="text-[9px] text-t3 text-right mt-1">
          +{Object.keys(counts).length - maxBars} more states
        </div>
      )}

      {/* Hover tooltip */}
      {hovered && counts[hovered] !== undefined && (
        <div
          className="mt-2 px-2 py-1.5 bg-surface border border-[var(--border)] rounded
                     text-[10px] font-mono text-t2"
        >
          |{hovered}⟩: {counts[hovered]} counts &nbsp; / &nbsp;
          {(counts[hovered] / shots * 100).toFixed(2)}% &nbsp; / &nbsp;
          P={(counts[hovered] / shots).toFixed(4)}
        </div>
      )}
    </div>
  );
}

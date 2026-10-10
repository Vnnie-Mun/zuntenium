// src/components/ai/FeatureImportanceBar.tsx
"use client";
import { memo, useState } from "react";

interface Props {
  name:        string;
  value:       number;   // 0–1
  maxValue?:   number;   // for normalisation (default 1)
  color?:      string;
  description?: string;
  rank?:       number;
}

export const FeatureImportanceBar = memo(function FeatureImportanceBar({
  name, value, maxValue = 1, color = "#00D4B8", description, rank,
}: Props) {
  const [hovered, setHovered] = useState(false);
  const pct = Math.round((value / maxValue) * 100);

  return (
    <div
      className="group relative"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-2 py-[3px]">
        {/* Rank */}
        {rank !== undefined && (
          <span className="text-[8px] text-t4 font-mono w-3 flex-shrink-0 text-right">
            {rank}
          </span>
        )}

        {/* Feature name */}
        <span
          className="text-[9px] font-mono flex-shrink-0 transition-colors duration-150"
          style={{
            width: 88,
            color: hovered ? "var(--t1)" : "var(--t2)",
          }}
        >
          {name}
        </span>

        {/* Bar */}
        <div
          className="flex-1 relative rounded-full overflow-hidden"
          style={{ height: 5, background: "var(--border,#162440)" }}
        >
          <div
            className="absolute left-0 top-0 h-full rounded-full transition-all duration-500"
            style={{
              width:      `${pct}%`,
              background: color,
              boxShadow:  hovered ? `0 0 6px ${color}80` : "none",
            }}
          />
        </div>

        {/* Value */}
        <span
          className="text-[9px] font-mono w-[30px] text-right flex-shrink-0"
          style={{ color: hovered ? color : "var(--t3)" }}
        >
          {value.toFixed(2)}
        </span>
      </div>

      {/* Tooltip */}
      {hovered && description && (
        <div
          className="absolute left-0 bottom-full mb-1.5 z-50
                     bg-card border border-[var(--border)] rounded
                     px-2 py-1.5 text-[9px] text-t2 leading-relaxed
                     pointer-events-none shadow-card whitespace-nowrap"
        >
          <span className="font-mono font-semibold" style={{ color }}>{name}</span>
          {" · "}{description}
        </div>
      )}
    </div>
  );
});

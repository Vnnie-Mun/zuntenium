// src/components/ai/ConfidenceIndicator.tsx
"use client";
import { memo } from "react";

interface Props {
  confidence: number;   // 0–1
  size?:      number;   // px
  label?:     string;
}

export const ConfidenceIndicator = memo(function ConfidenceIndicator({
  confidence, size = 72, label,
}: Props) {
  const R   = size * 0.42;
  const C   = 2 * Math.PI * R;
  const pct = Math.max(0, Math.min(1, confidence));
  const fill = C * pct;
  const pctLabel = Math.round(pct * 100);

  // Colour: ≥80% cyan, 60–80% blue, <60% amber
  const strokeColor = pct >= 0.8 ? "#00D4B8" : pct >= 0.6 ? "#0084FF" : "#F5A623";
  const textColor   = strokeColor;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
             style={{ transform: "rotate(-90deg)" }}>
          {/* Track */}
          <circle
            cx={size / 2} cy={size / 2} r={R}
            fill="none"
            stroke="var(--border,#162440)"
            strokeWidth={size * 0.09}
          />
          {/* Fill */}
          <circle
            cx={size / 2} cy={size / 2} r={R}
            fill="none"
            stroke={strokeColor}
            strokeWidth={size * 0.09}
            strokeDasharray={`${fill} ${C}`}
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 ${size * 0.06}px ${strokeColor}80)`,
              transition: "stroke-dasharray 0.6s ease",
            }}
          />
        </svg>
        {/* Center text */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center
                     font-mono font-bold leading-none"
          style={{ color: textColor }}
        >
          <span style={{ fontSize: size * 0.24 }}>{pctLabel}</span>
          <span style={{ fontSize: size * 0.12, opacity: 0.7 }}>%</span>
        </div>
      </div>
      {label && (
        <span className="text-[9px] text-t3 tracking-wide uppercase text-center">{label}</span>
      )}
    </div>
  );
});

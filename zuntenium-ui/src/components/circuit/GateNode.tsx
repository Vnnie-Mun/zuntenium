// src/components/circuit/GateNode.tsx
"use client";
import { memo } from "react";
import { GATE_META, formatAngle } from "@/lib/gates";
import type { UIGate } from "@/types";

interface Props {
  gate:       UIGate;
  isSelected: boolean;
  hasError:   boolean;
  onSelect:   (id: string) => void;
}

export const GateNode = memo(function GateNode({
  gate, isSelected, hasError, onSelect,
}: Props) {
  const meta  = GATE_META[gate.name];
  const label = meta.label;
  const color = meta.color;

  const borderColor = isSelected ? color           : `${color}40`;
  const bg          = meta.bg;
  const shadow      = isSelected
    ? `0 0 0 2px ${color}, 0 0 14px ${color}50`
    : hasError
    ? `0 0 0 1.5px #FF4757, 0 0 8px rgba(255,71,87,0.4)`
    : "0 2px 8px rgba(0,0,0,0.5)";

  return (
    <div
      data-gate-id={gate.id}
      onClick={e => { e.stopPropagation(); onSelect(gate.id); }}
      className="relative flex flex-col items-center justify-center cursor-pointer
                 select-none transition-transform duration-100 hover:scale-[1.07]"
      style={{
        width: 38, height: 38,
        borderRadius: 5,
        background: bg,
        border: `1.5px solid ${borderColor}`,
        boxShadow: shadow,
      }}
    >
      {/* Gate label */}
      <span
        className="font-mono text-[10px] font-bold leading-none tracking-tight"
        style={{ color }}
      >
        {label}
      </span>

      {/* Parameter display — Prompt 5 */}
      {meta.hasParams && gate.params.length > 0 && (
        <span
          className="font-mono text-[7px] mt-[2px] leading-none"
          style={{ color: `${color}CC` }}
        >
          {formatAngle(gate.params[0] ?? 0)}
        </span>
      )}

      {/* Error indicator — Prompt 4 */}
      {hasError && (
        <div
          className="absolute -top-1 -right-1 w-3 h-3 rounded-full
                     bg-danger text-white text-[7px] font-bold
                     flex items-center justify-center z-20"
          title="Gate has validation error"
        >
          !
        </div>
      )}
    </div>
  );
});

// src/components/layout/RightPanel.tsx
"use client";
import { useCircuitStore, useUIStore } from "@/store";
import { ParameterPanel }   from "@/components/circuit/ParameterPanel";
import { ExecutionPanel }   from "@/components/execution/ExecutionPanel";
import { ResultsPanel }     from "@/components/visualization/ResultsPanel";

const PANEL_TABS = [
  { id: "properties", label: "Properties" },
  { id: "execution",  label: "Execute"    },
  { id: "results",    label: "Results"    },
] as const;

type PanelId = typeof PANEL_TABS[number]["id"];

export function RightPanel() {
  const activePanel = useUIStore(s => s.activePanel);
  const setPanel    = useUIStore(s => s.setPanel);
  const selectedId  = useCircuitStore(s => s.selectedId);

  return (
    <aside
      className="w-[306px] flex-shrink-0 flex flex-col overflow-hidden
                 bg-surface border-l border-[var(--border)]"
    >
      {/* Tabs */}
      <div className="flex border-b border-[var(--border)] flex-shrink-0">
        {PANEL_TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setPanel(tab.id as PanelId)}
            className={`flex-1 h-[38px] border-none bg-transparent
                        text-[10px] font-bold tracking-[.5px] uppercase
                        cursor-pointer transition-colors duration-150 relative
                        ${activePanel === tab.id
                          ? "text-primary-DEFAULT"
                          : "text-t3 hover:text-t2"
                        }`}
          >
            {tab.label}
            {activePanel === tab.id && (
              <span
                className="absolute bottom-0 left-0 right-0 h-[2px]
                           bg-primary-DEFAULT"
              />
            )}
          </button>
        ))}
      </div>

      {/* Panel content */}
      <div className="flex-1 overflow-y-auto p-3.5">
        {activePanel === "properties" && (
          <ParameterPanel gateId={selectedId ?? ""} />
        )}
        {activePanel === "execution" && <ExecutionPanel />}
        {activePanel === "results"   && <ResultsPanel />}
      </div>
    </aside>
  );
}

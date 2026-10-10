// src/components/layout/TopBar.tsx
"use client";
import { useCircuitStore }    from "@/store";
import { useExecutionStore }  from "@/store";
import { useUIStore }         from "@/store";
import { uiCircuitToBackendJson } from "@/lib/gates";

const TABS = [
  { id: "composer",  label: "Circuit Composer" },
  { id: "jobs",      label: "Job Dashboard" },
  { id: "backends",  label: "Hardware" },
  { id: "ai",        label: "AI Insights" },
] as const;

export function TopBar() {
  const circuit    = useCircuitStore(s => s.circuit);
  const executing  = useExecutionStore(s => s.executing);
  const activeTab  = useUIStore(s => s.activeTab);
  const setTab     = useUIStore(s => s.setTab);
  const addToast   = useUIStore(s => s.addToast);
  const setPanel   = useUIStore(s => s.setPanel);

  function handleExport() {
    const json = JSON.stringify(uiCircuitToBackendJson(circuit), null, 2);
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    a.download = `${circuit.name || "circuit"}.json`;
    a.click();
    addToast({ title: "Exported", message: "Circuit JSON downloaded", type: "success" });
  }

  function handleRun() {
    setPanel("execution");
    // Execution is triggered inside ExecutionPanel
    window.dispatchEvent(new CustomEvent("zuntenium:run"));
  }

  return (
    <header
      className="h-[50px] flex-shrink-0 flex items-center gap-2.5 px-3.5
                 bg-surface border-b border-[var(--border)] relative z-50"
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 mr-2">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center
                     text-xs font-bold text-white"
          style={{ background: "linear-gradient(135deg,#0084FF,#00D4B8)" }}
        >
          Zt
        </div>
        <div>
          <div className="text-[13px] font-bold tracking-wide">ZUNTENIUM</div>
          <div className="text-[9px] text-t3 font-mono leading-none mt-0.5">
            Visual Tools Platform v5
          </div>
        </div>
      </div>

      <div className="w-px h-5 bg-[var(--border)] flex-shrink-0" />

      {/* Nav tabs */}
      {TABS.map(t => (
        <button
          key={t.id}
          onClick={() => setTab(t.id as typeof activeTab)}
          className={`h-[30px] px-3 rounded text-xs font-medium border transition-all duration-150
            ${activeTab === t.id
              ? "bg-primary-dim border-primary-DEFAULT/40 text-primary-DEFAULT"
              : "bg-transparent border-transparent text-t2 hover:bg-hover hover:text-t1"
            }`}
        >
          {t.label}
        </button>
      ))}

      <div className="flex-1" />

      {/* Status pills */}
      <div className="flex items-center gap-1.5 h-6 px-2.5 rounded-full
                      bg-[rgba(46,204,113,0.08)] border border-[rgba(46,204,113,0.2)]
                      text-[10px] font-mono text-t2">
        <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse-slow" />
        API Connected
      </div>

      {executing && (
        <div className="flex items-center gap-1.5 h-6 px-2.5 rounded-full
                        bg-[rgba(245,166,35,0.08)] border border-[rgba(245,166,35,0.2)]
                        text-[10px] font-mono text-warning">
          <span className="w-1.5 h-1.5 rounded-full bg-warning animate-spin-slow" />
          Running…
        </div>
      )}

      <button
        onClick={handleExport}
        className="btn text-[11px]"
      >
        ⬇ Export JSON
      </button>

      <button
        onClick={handleRun}
        disabled={executing}
        className={`btn text-[11px] ${executing ? "btn-running" : "bg-primary-DEFAULT border-primary-DEFAULT text-white hover:bg-[#006ACC]"}`}
      >
        {executing ? "⏳ Running…" : "▶ Run Circuit"}
      </button>

      <div
        className="w-7 h-7 rounded-full flex items-center justify-center
                   text-xs font-bold cursor-pointer"
        style={{ background: "linear-gradient(135deg,#1D4ED8,#7C3AED)" }}
      >
        QE
      </div>
    </header>
  );
}

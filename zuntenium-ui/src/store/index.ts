// src/store/index.ts
// ── Global Zustand stores ─────────────────────────────────────────────────
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import type {
  UICircuit, UIGate,
  JobResponse, JobResultResponse,
  PipelineStep, PipelineStepId, PipelineStepStatus,
  BackendChoice, OptimMode,
} from "@/types";
import { GATE_META } from "@/lib/gates";

// ────────────────────────────────────────────────────────────────────────────
// CIRCUIT STORE
// ────────────────────────────────────────────────────────────────────────────

interface CircuitState {
  circuit:    UICircuit;
  selectedId: string | null;
  history:    UICircuit[];
  historyIdx: number;

  // Actions
  setName:       (name: string) => void;
  setNumQubits:  (n: number) => void;
  addGate:       (gate: Omit<UIGate, "id">) => void;
  removeGate:    (id: string) => void;
  updateGate:    (id: string, patch: Partial<UIGate>) => void;
  moveGate:      (id: string, qubit: number, col: number) => void;
  clearCircuit:  () => void;
  setSelected:   (id: string | null) => void;
  undo:          () => void;
  redo:          () => void;
  loadCircuit:   (c: UICircuit) => void;
  getMaxCol:     () => number;
}

let gateCounter = 1;
const mkId = () => `g-${Date.now()}-${gateCounter++}`;

const DEFAULT_CIRCUIT: UICircuit = { name: "Bell State", numQubits: 2, gates: [] };

export const useCircuitStore = create<CircuitState>()(
  immer((set, get) => ({
    circuit:    { ...DEFAULT_CIRCUIT },
    selectedId: null,
    history:    [{ ...DEFAULT_CIRCUIT }],
    historyIdx: 0,

    setName: (name) => set(s => { s.circuit.name = name; }),

    setNumQubits: (n) => set(s => {
      const prev = { ...s.circuit };
      s.history.splice(s.historyIdx + 1);
      s.history.push(prev);
      s.historyIdx = s.history.length - 1;
      s.circuit.numQubits = Math.max(1, Math.min(200, n));
      // Remove gates on removed qubits
      s.circuit.gates = s.circuit.gates.filter(g => g.qubit < n);
    }),

    addGate: (gate) => set(s => {
      const prev = { ...s.circuit, gates: [...s.circuit.gates] };
      s.history.splice(s.historyIdx + 1);
      s.history.push(prev);
      s.historyIdx = s.history.length - 1;
      const meta = GATE_META[gate.name];
      const params = gate.params.length > 0
        ? gate.params
        : (meta.paramDefaults ?? []);
      s.circuit.gates.push({ ...gate, id: mkId(), params });
    }),

    removeGate: (id) => set(s => {
      const prev = { ...s.circuit, gates: [...s.circuit.gates] };
      s.history.splice(s.historyIdx + 1);
      s.history.push(prev);
      s.historyIdx = s.history.length - 1;
      s.circuit.gates = s.circuit.gates.filter(g => g.id !== id);
      if (s.selectedId === id) s.selectedId = null;
    }),

    updateGate: (id, patch) => set(s => {
      const g = s.circuit.gates.find(g => g.id === id);
      if (g) Object.assign(g, patch);
    }),

    moveGate: (id, qubit, col) => set(s => {
      const g = s.circuit.gates.find(g => g.id === id);
      if (g) { g.qubit = qubit; g.col = col; }
    }),

    clearCircuit: () => set(s => {
      const prev = { ...s.circuit, gates: [...s.circuit.gates] };
      s.history.splice(s.historyIdx + 1);
      s.history.push(prev);
      s.historyIdx = s.history.length - 1;
      s.circuit.gates = [];
      s.selectedId = null;
    }),

    setSelected: (id) => set(s => { s.selectedId = id; }),

    undo: () => set(s => {
      if (s.historyIdx > 0) {
        s.historyIdx--;
        s.circuit = { ...s.history[s.historyIdx]! };
      }
    }),

    redo: () => set(s => {
      if (s.historyIdx < s.history.length - 1) {
        s.historyIdx++;
        s.circuit = { ...s.history[s.historyIdx]! };
      }
    }),

    loadCircuit: (c) => set(s => {
      s.circuit    = c;
      s.selectedId = null;
      s.history    = [c];
      s.historyIdx = 0;
    }),

    getMaxCol: () => {
      const { gates } = get().circuit;
      return gates.reduce((m, g) => Math.max(m, g.col), -1);
    },
  }))
);

// ────────────────────────────────────────────────────────────────────────────
// EXECUTION STORE
// ────────────────────────────────────────────────────────────────────────────

const INITIAL_STEPS: PipelineStep[] = [
  { id:"circuit",    label:"Circuit",         detail:"Waiting…",                   status:"idle" },
  { id:"transpiler", label:"Transpiler",      detail:"Rule + RL passes",           status:"idle", badge:"RL",   badgeClass:"tag-ok"  },
  { id:"ai",         label:"AI Optimizer",    detail:"Feature extraction…",        status:"idle", badge:"AI",   badgeClass:"tag-ai"  },
  { id:"selector",   label:"Backend Selector",detail:"Scoring devices…",           status:"idle" },
  { id:"mhos",       label:"MHOS Orchestrator",detail:"IBM · IonQ · Braket…",     status:"idle", badge:"MHOS", badgeClass:"tag-mhos"},
  { id:"execution",  label:"Execution",       detail:"Running on device…",         status:"idle" },
  { id:"mitigation", label:"Error Mitigation",detail:"ZNE · PEC · Readout",       status:"idle" },
];

interface ExecutionState {
  steps:       PipelineStep[];
  executing:   boolean;
  currentJob?: JobResponse;
  result?:     JobResultResponse;
  jobHistory:  JobResponse[];

  // Config
  shots:              number;
  backend:            BackendChoice;
  optimMode:          OptimMode;
  mitigEnabled:       boolean;
  maxBondDim:         number;

  // Actions
  setStepStatus:  (id: PipelineStepId, status: PipelineStepStatus, detail?: string) => void;
  resetSteps:     () => void;
  setExecuting:   (v: boolean) => void;
  setCurrentJob:  (j: JobResponse | undefined) => void;
  setResult:      (r: JobResultResponse | undefined) => void;
  addJobToHistory:(j: JobResponse) => void;
  setShots:       (n: number) => void;
  setBackend:     (b: BackendChoice) => void;
  setOptimMode:   (m: OptimMode) => void;
  setMitigation:  (v: boolean) => void;
  setMaxBondDim:  (n: number) => void;
}

export const useExecutionStore = create<ExecutionState>()(
  immer(set => ({
    steps:       INITIAL_STEPS.map(s => ({ ...s })),
    executing:   false,
    jobHistory:  [],
    shots:       1024,
    backend:     "auto",
    optimMode:   "ai",
    mitigEnabled: false,
    maxBondDim:  64,

    setStepStatus: (id, status, detail) => set(s => {
      const step = s.steps.find(st => st.id === id);
      if (step) {
        step.status = status;
        if (detail !== undefined) step.detail = detail;
      }
    }),

    resetSteps: () => set(s => {
      s.steps = INITIAL_STEPS.map(st => ({ ...st }));
    }),

    setExecuting:    (v) => set(s => { s.executing = v; }),
    setCurrentJob:   (j) => set(s => { s.currentJob = j; }),
    setResult:       (r) => set(s => { s.result = r; }),
    addJobToHistory: (j) => set(s => { s.jobHistory.unshift(j); }),
    setShots:        (n) => set(s => { s.shots = n; }),
    setBackend:      (b) => set(s => { s.backend = b; }),
    setOptimMode:    (m) => set(s => { s.optimMode = m; }),
    setMitigation:   (v) => set(s => { s.mitigEnabled = v; }),
    setMaxBondDim:   (n) => set(s => { s.maxBondDim = n; }),
  }))
);

// ────────────────────────────────────────────────────────────────────────────
// UI STORE
// ────────────────────────────────────────────────────────────────────────────

type ActiveTab   = "composer" | "jobs" | "backends" | "ai";
type ActivePanel = "properties" | "execution" | "results";

interface UIState {
  activeTab:   ActiveTab;
  activePanel: ActivePanel;
  toasts:      Toast[];
  setTab:      (t: ActiveTab)   => void;
  setPanel:    (p: ActivePanel) => void;
  addToast:    (toast: Omit<Toast, "id">) => void;
  removeToast: (id: string) => void;
}

export interface Toast {
  id:      string;
  title:   string;
  message: string;
  type:    "success" | "error" | "warning" | "info";
}

export const useUIStore = create<UIState>()(
  immer(set => ({
    activeTab:   "composer",
    activePanel: "properties",
    toasts:      [],
    setTab:      (t) => set(s => { s.activeTab   = t; }),
    setPanel:    (p) => set(s => { s.activePanel = p; }),
    addToast: (toast) => set(s => {
      s.toasts.push({ ...toast, id: `toast-${Date.now()}` });
    }),
    removeToast: (id) => set(s => {
      s.toasts = s.toasts.filter(t => t.id !== id);
    }),
  }))
);

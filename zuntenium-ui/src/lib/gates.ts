// src/lib/gates.ts
// ── Gate metadata registry ────────────────────────────────────────────────
import type { GateMeta, GateName } from "@/types";

export const GATE_META: Record<GateName, GateMeta> = {
  H: {
    label: "H", fullName: "Hadamard", backendName: "h",
    description: "Creates equal superposition |+⟩ = (|0⟩+|1⟩)/√2",
    color: "#0084FF", bg: "rgba(0,132,255,0.22)",
    category: "single", hasParams: false,
  },
  X: {
    label: "X", fullName: "Pauli-X", backendName: "x",
    description: "Bit flip gate — maps |0⟩↔|1⟩",
    color: "#FF4757", bg: "rgba(255,71,87,0.22)",
    category: "single", hasParams: false,
  },
  Y: {
    label: "Y", fullName: "Pauli-Y", backendName: "y",
    description: "Combined bit and phase flip",
    color: "#FF6B35", bg: "rgba(255,107,53,0.22)",
    category: "single", hasParams: false,
  },
  Z: {
    label: "Z", fullName: "Pauli-Z", backendName: "z",
    description: "Phase flip — maps |1⟩ → −|1⟩",
    color: "#A855F7", bg: "rgba(168,85,247,0.22)",
    category: "single", hasParams: false,
  },
  S: {
    label: "S", fullName: "S Gate", backendName: "s",
    description: "Phase gate — π/2 rotation on Bloch sphere",
    color: "#00D4B8", bg: "rgba(0,212,184,0.22)",
    category: "single", hasParams: false,
  },
  T: {
    label: "T", fullName: "T Gate", backendName: "t",
    description: "T gate — π/4 phase rotation (non-Clifford)",
    color: "#F5A623", bg: "rgba(245,166,35,0.22)",
    category: "single", hasParams: false,
  },
  SDG: {
    label: "S†", fullName: "S-dagger", backendName: "sdg",
    description: "Inverse S gate — −π/2 phase",
    color: "#00D4B8", bg: "rgba(0,212,184,0.18)",
    category: "single", hasParams: false,
  },
  TDG: {
    label: "T†", fullName: "T-dagger", backendName: "tdg",
    description: "Inverse T gate — −π/4 phase",
    color: "#F5A623", bg: "rgba(245,166,35,0.18)",
    category: "single", hasParams: false,
  },
  ID: {
    label: "I", fullName: "Identity", backendName: "id",
    description: "Identity gate — no operation",
    color: "#4A6080", bg: "rgba(74,96,128,0.22)",
    category: "single", hasParams: false,
  },
  SX: {
    label: "√X", fullName: "√X Gate", backendName: "sx",
    description: "Square root of X — half bit-flip",
    color: "#FF4757", bg: "rgba(255,71,87,0.18)",
    category: "single", hasParams: false,
  },
  RX: {
    label: "Rx", fullName: "Rotation X", backendName: "rx",
    description: "Rotation about X-axis by θ radians",
    color: "#E91E8C", bg: "rgba(233,30,140,0.22)",
    category: "parametric", hasParams: true,
    paramNames: ["θ"], paramDefaults: [Math.PI / 2],
  },
  RY: {
    label: "Ry", fullName: "Rotation Y", backendName: "ry",
    description: "Rotation about Y-axis by θ radians",
    color: "#FF6B35", bg: "rgba(255,107,53,0.22)",
    category: "parametric", hasParams: true,
    paramNames: ["θ"], paramDefaults: [Math.PI / 2],
  },
  RZ: {
    label: "Rz", fullName: "Rotation Z", backendName: "rz",
    description: "Rotation about Z-axis by θ radians",
    color: "#7C3AED", bg: "rgba(124,58,237,0.22)",
    category: "parametric", hasParams: true,
    paramNames: ["θ"], paramDefaults: [Math.PI / 2],
  },
  P: {
    label: "P", fullName: "Phase", backendName: "p",
    description: "Phase rotation by λ radians",
    color: "#A855F7", bg: "rgba(168,85,247,0.22)",
    category: "parametric", hasParams: true,
    paramNames: ["λ"], paramDefaults: [Math.PI / 4],
  },
  CX: {
    label: "CX", fullName: "CNOT", backendName: "cx",
    description: "Controlled-NOT — entangling gate",
    color: "#0084FF", bg: "rgba(0,132,255,0.22)",
    category: "two-qubit", hasParams: false, isTwoQubit: true,
  },
  CZ: {
    label: "CZ", fullName: "Ctrl-Z", backendName: "cz",
    description: "Controlled-Z — phase kickback gate",
    color: "#00D4B8", bg: "rgba(0,212,184,0.22)",
    category: "two-qubit", hasParams: false, isTwoQubit: true,
  },
  CY: {
    label: "CY", fullName: "Ctrl-Y", backendName: "cy",
    description: "Controlled-Y gate",
    color: "#FF6B35", bg: "rgba(255,107,53,0.22)",
    category: "two-qubit", hasParams: false, isTwoQubit: true,
  },
  SWAP: {
    label: "SW", fullName: "SWAP", backendName: "swap",
    description: "Exchanges the states of two qubits",
    color: "#F59E0B", bg: "rgba(245,158,11,0.22)",
    category: "two-qubit", hasParams: false, isTwoQubit: true,
  },
  CP: {
    label: "CP", fullName: "Ctrl-Phase", backendName: "cp",
    description: "Controlled phase rotation",
    color: "#A855F7", bg: "rgba(168,85,247,0.22)",
    category: "two-qubit", hasParams: true,
    paramNames: ["λ"], paramDefaults: [Math.PI / 4], isTwoQubit: true,
  },
  CCX: {
    label: "CCX", fullName: "Toffoli", backendName: "ccx",
    description: "3-qubit CCNOT — universal reversible gate",
    color: "#10B981", bg: "rgba(16,185,129,0.22)",
    category: "multi-qubit", hasParams: false, isThreeQubit: true,
  },
};

export const GATE_CATEGORIES = {
  single:       ["H","X","Y","Z","S","T","SDG","TDG","SX","ID"] as GateName[],
  parametric:   ["RX","RY","RZ","P"] as GateName[],
  "two-qubit":  ["CX","CZ","CY","SWAP","CP"] as GateName[],
  "multi-qubit":["CCX"] as GateName[],
};

export const CATEGORY_LABELS: Record<string, string> = {
  single:       "Single Qubit",
  parametric:   "Parametric",
  "two-qubit":  "Two Qubit",
  "multi-qubit":"Multi-Qubit",
};

// ── Angle formatting ────────────────────────────────────────────────────────
const PI_FRACS: Array<[number, string]> = [
  [0,           "0"],
  [Math.PI/6,   "π/6"],
  [Math.PI/4,   "π/4"],
  [Math.PI/3,   "π/3"],
  [Math.PI/2,   "π/2"],
  [Math.PI*2/3, "2π/3"],
  [Math.PI*3/4, "3π/4"],
  [Math.PI,     "π"],
  [Math.PI*5/4, "5π/4"],
  [Math.PI*3/2, "3π/2"],
  [Math.PI*7/4, "7π/4"],
  [Math.PI*2,   "2π"],
  [-Math.PI/6,  "-π/6"],
  [-Math.PI/4,  "-π/4"],
  [-Math.PI/3,  "-π/3"],
  [-Math.PI/2,  "-π/2"],
  [-Math.PI,    "-π"],
];

export function formatAngle(val: number): string {
  for (const [n, s] of PI_FRACS) {
    if (Math.abs(val - n) < 0.01) return s;
  }
  return val.toFixed(3);
}

export function parseAngle(s: string): number | null {
  const cleaned = s.trim().replace(/π/g, `${Math.PI}`);
  try {
    // Safe numeric evaluation for expressions like "PI/2"
    const result = Function(`"use strict"; return (${cleaned})`)() as number;
    return isFinite(result) ? result : null;
  } catch {
    return null;
  }
}

export const ANGLE_PRESETS = [
  { label: "0",   value: 0 },
  { label: "π/4", value: Math.PI / 4 },
  { label: "π/2", value: Math.PI / 2 },
  { label: "π",   value: Math.PI },
  { label: "3π/2",value: Math.PI * 3/2 },
  { label: "2π",  value: Math.PI * 2 },
  { label: "-π/2",value: -Math.PI / 2 },
  { label: "-π",  value: -Math.PI },
];

// ── Circuit serialisation → backend JSON ─────────────────────────────────
import type { UIGate, UICircuit, CircuitSchema } from "@/types";

export function uiCircuitToBackendJson(circuit: UICircuit): CircuitSchema {
  const sorted = [...circuit.gates].sort((a, b) =>
    a.col !== b.col ? a.col - b.col : a.qubit - b.qubit
  );

  const gates = sorted.map(g => {
    const meta = GATE_META[g.name];
    const qubits: number[] = [g.qubit];
    if (meta.isTwoQubit && g.target !== undefined) qubits.push(g.target);
    if (meta.isThreeQubit) {
      if (g.target  !== undefined) qubits.push(g.target);
      if (g.target2 !== undefined) qubits.push(g.target2);
    }
    return {
      name:   meta.backendName,
      qubits,
      params: g.params,
    };
  });

  return {
    num_qubits: circuit.numQubits,
    name:       circuit.name,
    gates,
  };
}

export function backendJsonToUICircuit(json: CircuitSchema): UICircuit {
  const gates: UIGate[] = json.gates.map((g, idx) => {
    const gateName = Object.entries(GATE_META)
      .find(([, m]) => m.backendName === g.name.toLowerCase())?.[0] as GateName | undefined;
    const name = gateName ?? ("H" as GateName);
    const meta = GATE_META[name];
    return {
      id:      `imported-${idx}`,
      name,
      qubit:   g.qubits[0] ?? 0,
      target:  meta.isTwoQubit    ? g.qubits[1] : undefined,
      target2: meta.isThreeQubit  ? g.qubits[2] : undefined,
      col:     idx,
      params:  g.params,
    };
  });
  return { name: json.name ?? "Imported", numQubits: json.num_qubits, gates };
}

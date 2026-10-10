export const GATE_LIBRARY = [
  { type: 'H', label: 'Hadamard', symbol: 'H', qubits: 1, group: 'Single qubit' },
  { type: 'X', label: 'Pauli X', symbol: 'X', qubits: 1, group: 'Single qubit' },
  { type: 'Y', label: 'Pauli Y', symbol: 'Y', qubits: 1, group: 'Single qubit' },
  { type: 'Z', label: 'Pauli Z', symbol: 'Z', qubits: 1, group: 'Single qubit' },
  { type: 'S', label: 'Phase S', symbol: 'S', qubits: 1, group: 'Single qubit' },
  { type: 'T', label: 'Phase T', symbol: 'T', qubits: 1, group: 'Single qubit' },
  { type: 'RX', label: 'Rotation X', symbol: 'Rx', qubits: 1, group: 'Rotations' },
  { type: 'RY', label: 'Rotation Y', symbol: 'Ry', qubits: 1, group: 'Rotations' },
  { type: 'RZ', label: 'Rotation Z', symbol: 'Rz', qubits: 1, group: 'Rotations' },
  { type: 'CNOT', label: 'Controlled X', symbol: 'CX', qubits: 2, group: 'Controlled' },
  { type: 'CZ', label: 'Controlled Z', symbol: 'CZ', qubits: 2, group: 'Controlled' },
  { type: 'SWAP', label: 'Swap', symbol: 'SW', qubits: 2, group: 'Controlled' },
  { type: 'CCX', label: 'Toffoli', symbol: 'CCX', qubits: 3, group: 'Controlled' },
  { type: 'MEASURE', label: 'Measure', symbol: 'M', qubits: 1, group: 'Measurement' },
] as const;
export type GateType = (typeof GATE_LIBRARY)[number]['type'];
export type NoiseModel = 'none' | 'bitflip' | 'phaseflip' | 'depolarizing' | 'amplitude_damping';
export interface Gate {
  id: string;
  type: GateType;
  column: number;
  controls: number[];
  targets: number[];
  params: { theta?: number };
  classicalControl?: ClassicalControl;
}
/** Single bit `{qubit, equals: 0|1}` or register `{qubits (LSB first), equals}`. */
export interface ClassicalControl { qubit?: number; qubits?: number[]; equals: number }
export function conditionQubits(c: ClassicalControl): number[] { return c.qubits ?? (c.qubit === undefined ? [] : [c.qubit]); }
export function describeCondition(c: ClassicalControl): string {
  const qs = conditionQubits(c);
  return qs.length === 1 ? `c${qs[0]} == ${c.equals}` : `[${[...qs].reverse().map(q => `c${q}`).join(' ')}] == ${c.equals}`;
}
export interface Circuit { version: 1; numQubits: number; gates: Gate[] }
export interface StateEntry { index: number; label: string; amplitude: { re: number; im: number }; probability: number; phase: number }
export interface BlochVector { qubit: number; x: number; y: number; z: number }
export interface RunResult {
  backend: 'local' | 'zuntenium'; backendName?: string; numQubits: number; statevector: StateEntry[]; statevectorTruncated: boolean; totalNonzeroStates: number;
  blochVectors: BlochVector[]; counts: Record<string, number>; shots: number;
  noise: { model: NoiseModel; p: number } | null; midCircuitMeasurement: boolean;
  hasClassicalControl: boolean; noiseShotsCapped: boolean; executionTimeMs: number; seed: number | null;
}
export const noiseOptions: { id: NoiseModel; label: string }[] = [
  { id: 'none', label: 'Ideal' }, { id: 'bitflip', label: 'Bit flip' },
  { id: 'phaseflip', label: 'Phase flip' }, { id: 'depolarizing', label: 'Depolarizing' },
  { id: 'amplitude_damping', label: 'Amplitude damping' },
];
export function touched(g: Gate) { return [...g.controls, ...g.targets]; }
export function validateCircuit(c: Circuit) {
  if (!Number.isInteger(c.numQubits) || c.numQubits < 1 || c.numQubits > 12) throw new Error('Choose between 1 and 12 qubits.');
  const occupied = new Set<string>();
  for (const g of c.gates) {
    const definition = GATE_LIBRARY.find(item => item.type === g.type);
    if (!definition || !Number.isInteger(g.column) || g.column < 0 || g.column > 255) throw new Error('Circuit contains an invalid gate or column.');
    const qs = touched(g);
    if (qs.length !== definition.qubits || new Set(qs).size !== qs.length || qs.some(q => !Number.isInteger(q) || q < 0 || q >= c.numQubits)) throw new Error(`Invalid qubits for ${g.type}.`);
    for (const q of qs) {
      const key = `${g.column}:${q}`;
      if (occupied.has(key)) throw new Error(`Two gates overlap at q${q}, step ${g.column + 1}.`);
      occupied.add(key);
    }
    if (['RX','RY','RZ'].includes(g.type) && !Number.isFinite(g.params.theta)) throw new Error(`${g.type} requires a finite angle.`);
    if (g.classicalControl) {
      const qs = conditionQubits(g.classicalControl); const { equals } = g.classicalControl;
      if (g.type === 'MEASURE') throw new Error('Measurements cannot be conditional.');
      if (!qs.length || qs.length > 8 || new Set(qs).size !== qs.length || qs.some(q => !Number.isInteger(q) || q < 0 || q >= c.numQubits)) throw new Error('A condition must use 1–8 distinct measured bits.');
      if (!Number.isInteger(equals) || equals < 0 || equals >= 2 ** qs.length) throw new Error(`Condition value must be between 0 and ${2 ** qs.length - 1}.`);
      if (qs.some(q => !c.gates.some(m => m.type === 'MEASURE' && m.targets[0] === q && m.column < g.column))) throw new Error('A condition needs an earlier measurement of every bit it uses.');
    }
  }
}
export function makeGate(type: GateType, qubits: number[], column: number, theta?: number): Gate {
  const controls = type === 'CCX' ? qubits.slice(0, 2) : ['CNOT','CZ'].includes(type) ? qubits.slice(0, 1) : [];
  const targets = controls.length ? qubits.slice(controls.length) : qubits;
  const id = globalThis.crypto?.randomUUID?.()
    ?? `gate-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return { id, type, column, controls, targets, params: theta === undefined ? {} : { theta } };
}

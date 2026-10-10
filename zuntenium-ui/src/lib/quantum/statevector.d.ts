import type { Gate, StateEntry, BlochVector, NoiseModel } from './model';
export const MAX_QUBITS: number;
export class StateVector {
  constructor(numQubits: number, random?: () => number);
  n: number; size: number; re: Float64Array; im: Float64Array;
  applyGate(gate: Gate): void;
  applyAmplitudeDamping(target: number, gamma: number): void;
  run(gates: Gate[]): void;
  reset(): void;
  runTrajectory(gates: Gate[], noise: { model: NoiseModel; p: number } | null): string;
  probabilities(): Float64Array;
  toStatevectorJSON(input?: { maxEntries?: number; probs?: Float64Array }): { entries: StateEntry[]; truncated: boolean; totalNonzero: number };
  blochVectors(): BlochVector[];
  sample(shots: number, probs?: Float64Array): Record<string, number>;
}
export function hasMidCircuitMeasurement(gates: Gate[]): boolean;
export function hasClassicalControl(gates: Gate[]): boolean;

import { StateVector, hasClassicalControl, hasMidCircuitMeasurement } from './statevector.js';
import { validateCircuit, type Circuit, type NoiseModel, type RunResult } from './model';

export function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
export function runCircuit(circuit: Circuit, shots: number, model: NoiseModel, p: number, seed: number | null): RunResult {
  validateCircuit(circuit);
  if (!Number.isInteger(shots) || shots < 1 || shots > 20000) throw new Error('Shots must be between 1 and 20,000.');
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error('Noise probability must be between 0 and 1.');
  if (seed !== null && (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295)) throw new Error('Seed must be a whole number from 0 to 4,294,967,295.');
  const started = performance.now();
  const rng = seed === null ? Math.random : seededRandom(seed);
  const ideal = new StateVector(circuit.numQubits, rng);
  ideal.run(circuit.gates);
  const probs = ideal.probabilities();
  const { entries, truncated, totalNonzero } = ideal.toStatevectorJSON({ probs });
  const dynamic = model !== 'none' || hasMidCircuitMeasurement(circuit.gates) || hasClassicalControl(circuit.gates);
  const actualShots = dynamic ? Math.min(shots, 5000) : shots;
  const noise = model === 'none' ? null : { model, p };
  const counts: Record<string, number> = {};
  if (dynamic) {
    const trial = new StateVector(circuit.numQubits, rng);
    for (let i = 0; i < actualShots; i++) {
      trial.reset();
      const label = trial.runTrajectory(circuit.gates, noise);
      counts[label] = (counts[label] ?? 0) + 1;
    }
  } else Object.assign(counts, ideal.sample(actualShots, probs));
  return { backend: 'local', numQubits: circuit.numQubits, statevector: entries, statevectorTruncated: truncated,
    totalNonzeroStates: totalNonzero, blochVectors: ideal.blochVectors(), counts, shots: actualShots,
    noise, midCircuitMeasurement: hasMidCircuitMeasurement(circuit.gates), hasClassicalControl: hasClassicalControl(circuit.gates),
    noiseShotsCapped: dynamic && actualShots < shots, executionTimeMs: Math.round(performance.now() - started), seed };
}

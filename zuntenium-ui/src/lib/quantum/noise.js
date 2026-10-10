// Single-qubit noise channels, applied as a stochastic Pauli error
// immediately after the gate that touched that qubit. Called once per qubit
// per gate from StateVector.runWithNoise — see the comment there for why
// this (quantum-trajectory / Monte Carlo wavefunction) approach is used
// instead of full density-matrix simulation.
export function applyNoiseToQubit(sv, qubit, model, p, random = Math.random) {
  if (!model || model === 'none' || p <= 0) return;
  const r = random();

  if (model === 'bitflip') {
    if (r < p) sv.applyGate({ type: 'X', targets: [qubit] });
    return;
  }
  if (model === 'phaseflip') {
    if (r < p) sv.applyGate({ type: 'Z', targets: [qubit] });
    return;
  }
  if (model === 'depolarizing') {
    // With probability p, one of X/Y/Z is applied with equal likelihood —
    // the standard single-qubit depolarizing-channel unraveling.
    if (r < p) {
      const pauli = ['X', 'Y', 'Z'][Math.floor(random() * 3)];
      sv.applyGate({ type: pauli, targets: [qubit] });
    }
    return;
  }
  if (model === 'amplitude_damping') {
    sv.applyAmplitudeDamping(qubit, p);
    return;
  }
  throw new Error(`Unknown noise model: ${model}`);
}

export const NOISE_MODELS = [
  { id: 'none', label: 'None (ideal)' },
  { id: 'bitflip', label: 'Bit-flip' },
  { id: 'phaseflip', label: 'Phase-flip' },
  { id: 'depolarizing', label: 'Depolarizing' },
  { id: 'amplitude_damping', label: 'Amplitude damping (T1-style)' },
];

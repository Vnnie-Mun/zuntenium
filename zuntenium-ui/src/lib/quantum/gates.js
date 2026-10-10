import { C, cExpI } from './complex.js';

const SQRT1_2 = Math.SQRT1_2;

// Every matrix is [[a, b], [c, d]] applied as [a b; c d] * [amp0; amp1],
// where amp0 is the amplitude with the acted-on qubit = 0 and amp1 = 1.
// These are the standard textbook/Qiskit definitions — check against any
// reference (e.g. Nielsen & Chuang) if you want to verify independently.
export function baseMatrix(type, params = {}) {
  switch (type) {
    case 'I':
      return [[C(1), C(0)], [C(0), C(1)]];
    case 'H':
      return [[C(SQRT1_2), C(SQRT1_2)], [C(SQRT1_2), C(-SQRT1_2)]];
    case 'X':
      return [[C(0), C(1)], [C(1), C(0)]];
    case 'Y':
      return [[C(0), C(0, -1)], [C(0, 1), C(0)]];
    case 'Z':
      return [[C(1), C(0)], [C(0), C(-1)]];
    case 'S':
      return [[C(1), C(0)], [C(0), C(0, 1)]];
    case 'SDG':
      return [[C(1), C(0)], [C(0), C(0, -1)]];
    case 'T':
      return [[C(1), C(0)], [C(0), cExpI(Math.PI / 4)]];
    case 'TDG':
      return [[C(1), C(0)], [C(0), cExpI(-Math.PI / 4)]];
    case 'RX': {
      const t = params.theta || 0;
      return [
        [C(Math.cos(t / 2)), C(0, -Math.sin(t / 2))],
        [C(0, -Math.sin(t / 2)), C(Math.cos(t / 2))],
      ];
    }
    case 'RY': {
      const t = params.theta || 0;
      return [
        [C(Math.cos(t / 2)), C(-Math.sin(t / 2))],
        [C(Math.sin(t / 2)), C(Math.cos(t / 2))],
      ];
    }
    case 'RZ': {
      const t = params.theta || 0;
      return [
        [cExpI(-t / 2), C(0)],
        [C(0), cExpI(t / 2)],
      ];
    }
    default:
      throw new Error(`Unknown gate type: ${type}`);
  }
}

// Gate types whose "controlled" behaviour is just a controlled version of a
// single-qubit base matrix. SWAP and MEASURE are handled separately since
// they don't fit this shape.
export const CONTROLLED_BASE = {
  CNOT: 'X',
  CX: 'X',
  CY: 'Y',
  CZ: 'Z',
  CCX: 'X',
  TOFFOLI: 'X',
};

export const GATE_LIBRARY = [
  { type: 'H', label: 'H', qubits: 1, group: 'Single-qubit' },
  { type: 'X', label: 'X', qubits: 1, group: 'Single-qubit' },
  { type: 'Y', label: 'Y', qubits: 1, group: 'Single-qubit' },
  { type: 'Z', label: 'Z', qubits: 1, group: 'Single-qubit' },
  { type: 'S', label: 'S', qubits: 1, group: 'Single-qubit' },
  { type: 'T', label: 'T', qubits: 1, group: 'Single-qubit' },
  { type: 'RX', label: 'Rx(θ)', qubits: 1, group: 'Parameterized', needsAngle: true },
  { type: 'RY', label: 'Ry(θ)', qubits: 1, group: 'Parameterized', needsAngle: true },
  { type: 'RZ', label: 'Rz(θ)', qubits: 1, group: 'Parameterized', needsAngle: true },
  { type: 'CNOT', label: 'CNOT', qubits: 2, group: 'Multi-qubit' },
  { type: 'CZ', label: 'CZ', qubits: 2, group: 'Multi-qubit' },
  { type: 'SWAP', label: 'SWAP', qubits: 2, group: 'Multi-qubit' },
  { type: 'CCX', label: 'Toffoli', qubits: 3, group: 'Multi-qubit' },
  { type: 'MEASURE', label: 'Measure', qubits: 1, group: 'Measurement' },
];

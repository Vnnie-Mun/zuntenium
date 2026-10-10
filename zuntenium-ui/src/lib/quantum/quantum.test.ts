import { describe, expect, it } from 'vitest';
import { makeGate, type Circuit } from './model';
import { parseQasm, gatesToQasm } from './qasm';
import { runCircuit, seededRandom } from './run';
import { StateVector } from './statevector.js';

const circuit = (numQubits: number, gates: Circuit['gates']): Circuit => ({ version:1, numQubits, gates });

describe('quantum simulator', () => {
  it('creates the Bell state with the correct amplitudes and deterministic shots', () => {
    const bell = circuit(2,[makeGate('H',[0],0),makeGate('CNOT',[0,1],1)]);
    const result = runCircuit(bell,1000,'none',0,42);
    const probabilities = Object.fromEntries(result.statevector.map(entry => [entry.label,entry.probability]));
    expect(probabilities['00']).toBeCloseTo(.5,6);
    expect(probabilities['11']).toBeCloseTo(.5,6);
    expect(Object.keys(result.counts).sort()).toEqual(['00','11']);
    expect(Object.values(result.counts).reduce((sum,value) => sum+value,0)).toBe(1000);
  });

  it('creates a three-qubit GHZ state', () => {
    const ghz = circuit(3,[makeGate('H',[0],0),makeGate('CNOT',[0,1],1),makeGate('CNOT',[1,2],2)]);
    const result = runCircuit(ghz,100,'none',0,7);
    expect(Object.fromEntries(result.statevector.map(entry => [entry.label,entry.probability]))).toMatchObject({'000':.5,'111':.5});
  });

  it('applies mid-circuit measurement and classical feedback per trajectory', () => {
    const feedback = circuit(1,[makeGate('H',[0],0),makeGate('MEASURE',[0],1),{...makeGate('X',[0],2),classicalControl:{qubit:0,equals:1}}]);
    expect(runCircuit(feedback,500,'none',0,12).counts).toEqual({'0':500});
  });

  it('amplitude damping maps an excited qubit to ground when gamma is one', () => {
    const sv = new StateVector(1,seededRandom(5));
    sv.applyGate(makeGate('X',[0],0));
    sv.applyAmplitudeDamping(0,1);
    expect(Array.from(sv.probabilities())).toEqual([1,0]);
  });
});

describe('OpenQASM 2 exchange', () => {
  it('round-trips every supported operation and a classical condition', () => {
    const original = circuit(3,[
      makeGate('H',[0],0), makeGate('RX',[1],1,Math.PI/2), makeGate('CNOT',[0,2],2),
      makeGate('SWAP',[1,2],3), makeGate('MEASURE',[0],4),
      {...makeGate('X',[2],5),classicalControl:{qubit:0,equals:1}},
    ]);
    const restored = parseQasm(gatesToQasm(original.numQubits,original.gates));
    expect(restored.numQubits).toBe(3);
    expect(restored.gates.map(g => ({type:g.type,controls:g.controls,targets:g.targets,theta:g.params.theta,condition:g.classicalControl})))
      .toEqual(original.gates.map(g => ({type:g.type,controls:g.controls,targets:g.targets,theta:g.params.theta,condition:g.classicalControl})));
  });

  it('reports source lines for unsupported input', () => {
    expect(() => parseQasm('OPENQASM 2.0;\nqreg q[2];\nu3(1,2,3) q[0];')).toThrow(/Line 3/);
  });
});
describe('circuit validation guards', () => {
  it('rejects overlapping gates and out-of-range qubits', async () => {
    const { validateCircuit } = await import('./model');
    expect(() => validateCircuit(circuit(2,[makeGate('H',[0],0),makeGate('X',[0],0)]))).toThrow();
    expect(() => validateCircuit(circuit(2,[makeGate('H',[3],0)]))).toThrow();
  });
  it('rejects conditions without an earlier measurement', async () => {
    const { validateCircuit } = await import('./model');
    const g = { ...makeGate('X',[1],1), classicalControl:{ qubit:0, equals:1 as const } };
    expect(() => validateCircuit(circuit(2,[g]))).toThrow();
    expect(() => validateCircuit(circuit(2,[makeGate('MEASURE',[0],0),g]))).not.toThrow();
  });
  it('probabilities always sum to one and seeds repeat', () => {
    const c = circuit(3,[makeGate('H',[0],0),makeGate('RY',[1],0,0.7),makeGate('CCX',[0,1,2],1)]);
    const a = runCircuit(c,500,'depolarizing',0.05,7), b = runCircuit(c,500,'depolarizing',0.05,7);
    expect(a.statevector.reduce((s,e) => s+e.probability,0)).toBeCloseTo(1,9);
    expect(a.counts).toEqual(b.counts);
  });
});

describe('multi-bit classical control', () => {
  // Prepare q0=1, q1=1 deterministically, measure both, flip q2 only if register == 3.
  const build = (equals: number) => circuit(3,[
    makeGate('X',[0],0), makeGate('X',[1],0),
    makeGate('MEASURE',[0],1), makeGate('MEASURE',[1],1),
    { ...makeGate('X',[2],2), classicalControl:{ qubits:[0,1], equals } },
  ]);
  it('fires only when the register value matches', () => {
    expect(Object.keys(runCircuit(build(3),50,'none',0,1).counts)).toEqual(['111']);
    expect(Object.keys(runCircuit(build(2),50,'none',0,1).counts)).toEqual(['011']);
  });
  it('rejects values that do not fit the register', async () => {
    const { validateCircuit } = await import('./model');
    expect(() => validateCircuit(build(4))).toThrow();
  });
  it('round-trips through OpenQASM', () => {
    const c = build(3);
    const back = parseQasm(gatesToQasm(c.numQubits,c.gates));
    expect(back.gates.filter(g => g.type === 'MEASURE')).toHaveLength(2);
    expect(back.gates.find(g => g.classicalControl)?.classicalControl).toEqual({ qubits:[0,1], equals:3 });
    expect(runCircuit(back,50,'none',0,1).counts).toEqual(runCircuit(c,50,'none',0,1).counts);
  });
});

import { runCircuit } from './run';
import type { Circuit, NoiseModel } from './model';
self.onmessage = (event: MessageEvent<{ circuit: Circuit; shots: number; model: NoiseModel; p: number; seed: number | null }>) => {
  try { self.postMessage({ result: runCircuit(event.data.circuit, event.data.shots, event.data.model, event.data.p, event.data.seed) }); }
  catch (error) { self.postMessage({ error: error instanceof Error ? error.message : 'Simulation failed.' }); }
};

import { GATE_LIBRARY, makeGate, validateCircuit, type Circuit, type GateType } from './model';
import { gatesToQasm } from './qasmExport.js';
export { gatesToQasm };
const opMap: Record<string, GateType> = { h:'H',x:'X',y:'Y',z:'Z',s:'S',t:'T',rx:'RX',ry:'RY',rz:'RZ',cx:'CNOT',cz:'CZ',swap:'SWAP',ccx:'CCX' };
function angleValue(raw: string, line: number): number {
  const compact = raw.replace(/\s/g, '');
  const match = /^(-?)(?:(\d+(?:\.\d+)?|\.\d+)(\*)?)?(pi)?(?:\/(\d+(?:\.\d+)?))?$/.exec(compact);
  if (!match || (!match[2] && !match[4])) throw new Error(`Line ${line}: invalid rotation angle “${raw}”. Use a number or a multiple of pi.`);
  const value = (match[2] ? Number(match[2]) : 1) * (match[4] ? Math.PI : 1) / (match[5] ? Number(match[5]) : 1);
  if (!Number.isFinite(value)) throw new Error(`Line ${line}: rotation angle must be finite.`);
  return match[1] ? -value : value;
}
export function parseQasm(source: string): Circuit {
  if (source.length > 100000) throw new Error('QASM file is too large.');
  const lines = source.replace(/\/\/[^\n]*/g, '').split('\n');
  let numQubits = 0; let seenHeader = false;
  const registers = new Map<string, number>();
  const measured = new Map<string, number>();
  const gates: Circuit['gates'] = [];
  const parseRef = (raw: string, register: string, line: number) => {
    const ref = /^([a-zA-Z][\w]*)\[(\d+)\]$/.exec(raw.trim());
    if (!ref || ref[1] !== register || Number(ref[2]) >= (register === 'q' ? numQubits : (registers.get(register) ?? 0))) throw new Error(`Line ${line}: invalid ${register} register reference “${raw}”.`);
    return Number(ref[2]);
  };
  lines.forEach((raw, index) => {
    const line = index + 1; const text = raw.trim();
    if (!text) return;
    if (/^OPENQASM\s+2\.0\s*;$/i.test(text)) { seenHeader = true; return; }
    if (/^include\s+"qelib1\.inc"\s*;$/i.test(text)) return;
    const qreg = /^qreg\s+q\[(\d+)\]\s*;$/i.exec(text);
    if (qreg) { if (numQubits) throw new Error(`Line ${line}: only one quantum register is supported.`); numQubits = Number(qreg[1]); return; }
    const creg = /^creg\s+([a-zA-Z]\w*)\[(\d+)\]\s*;$/i.exec(text);
    if (creg) { if (registers.has(creg[1] ?? '')) throw new Error(`Line ${line}: duplicate classical register.`); registers.set(creg[1] ?? '', Number(creg[2])); return; }
    const measurement = /^measure\s+([^\s]+)\s*->\s*([^\s]+)\s*;$/i.exec(text);
    if (measurement) {
      const q = parseRef(measurement[1] ?? '', 'q', line);
      const target = /^([a-zA-Z]\w*)\[(\d+)\]$/.exec(measurement[2] ?? '');
      if (!target || !registers.has(target[1] ?? '') || Number(target[2]) >= (registers.get(target[1] ?? '') ?? 0)) throw new Error(`Line ${line}: invalid measurement destination.`);
      measured.set(`${target[1]}:${target[2]}`, q);
      // Consecutive re-measurement of the same qubit only records another classical copy.
      const prev = gates[gates.length - 1];
      if (prev?.type === 'MEASURE' && prev.targets[0] === q) return;
      gates.push(makeGate('MEASURE',[q],gates.length)); return;
    }
    const operation = /^(?:if\s*\(\s*([a-zA-Z]\w*)\s*==\s*(\d+)\s*\)\s*)?([a-z]+)(?:\s*\(([^)]+)\))?\s+(.+?)\s*;$/i.exec(text);
    if (!operation) throw new Error(`Line ${line}: unsupported or malformed statement “${text}”.`);
    const type = opMap[(operation[3] ?? '').toLowerCase()];
    if (!type) throw new Error(`Line ${line}: unsupported gate “${operation[3]}”.`);
    const qs = (operation[5] ?? '').split(',').map(ref => parseRef(ref,'q',line));
    const definition = GATE_LIBRARY.find(g => g.type === type);
    if (qs.length !== definition?.qubits) throw new Error(`Line ${line}: ${type} needs ${definition?.qubits} qubit(s).`);
    const rotation = ['RX','RY','RZ'].includes(type);
    if (rotation !== Boolean(operation[4])) throw new Error(`Line ${line}: ${type} ${rotation ? 'needs' : 'does not accept'} an angle.`);
    const gate = makeGate(type, qs, gates.length, rotation ? angleValue(operation[4] ?? '', line) : undefined);
    if (operation[1]) {
      const register = operation[1]; const size = registers.get(register);
      const value = Number(operation[2]);
      if (!size) throw new Error(`Line ${line}: unknown classical register “${register}”.`);
      if (size > 8) throw new Error(`Line ${line}: conditions support registers of up to 8 bits.`);
      if (value >= 2 ** size) throw new Error(`Line ${line}: value ${value} does not fit in ${size}-bit register ${register}.`);
      const qs = Array.from({ length: size }, (_, i) => measured.get(`${register}:${i}`));
      if (qs.some(q => q === undefined)) throw new Error(`Line ${line}: condition references a bit of ${register} that has not been measured.`);
      gate.classicalControl = size === 1 ? { qubit: qs[0] as number, equals: value } : { qubits: qs as number[], equals: value };
    }
    gates.push(gate);
  });
  if (!seenHeader || !numQubits) throw new Error('OpenQASM 2.0 header and qreg q[n] are required.');
  const circuit: Circuit = { version:1, numQubits, gates };
  validateCircuit(circuit);
  return circuit;
}

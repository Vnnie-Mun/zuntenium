// Serializes our gate representation to OpenQASM 2.0 so a circuit built
// here can be opened in Qiskit or any other tool that reads QASM. This is a
// pure, backend-independent transform of the same gate objects the
// simulator already consumes — no new data model.
//
// Classical registers: OpenQASM 2.0's `if(reg==N)` compares an ENTIRE
// classical register to an integer, not a single bit within a shared one —
// `if(c[0]==1)` is not valid QASM. So each qubit gets its own single-bit
// register (c0, c1, ...), making `if(cN==1)` unambiguous.
const QASM_OP = {
  H: 'h', X: 'x', Y: 'y', Z: 'z', S: 's', T: 't',
  CNOT: 'cx', CZ: 'cz', SWAP: 'swap', CCX: 'ccx',
};

export function gatesToQasm(numQubits, gates) {
  const lines = [
    'OPENQASM 2.0;',
    'include "qelib1.inc";',
    `qreg q[${numQubits}];`,
  ];
  for (let q = 0; q < numQubits; q++) lines.push(`creg c${q}[1];`);
  // Multi-bit conditions: one extra register per distinct bit set (LSB first).
  // Each measurement is also recorded into every register containing that
  // qubit; a repeated projective measurement returns the same outcome.
  const regs = new Map();
  for (const g of gates) {
    const qs = g.classicalControl?.qubits;
    if (qs && qs.length > 1 && !regs.has(qs.join(','))) regs.set(qs.join(','), { name: `r${regs.size}`, qubits: qs });
  }
  for (const r of regs.values()) lines.push(`creg ${r.name}[${r.qubits.length}];`);
  lines.push('');

  const sorted = [...gates].sort((a, b) => (a.column ?? 0) - (b.column ?? 0));

  for (const g of sorted) {
    const controls = g.controls || [];
    const targets = g.targets || [];

    if (g.type === 'MEASURE') {
      const q = targets[0];
      lines.push(`measure q[${q}] -> c${q}[0];`);
      for (const r of regs.values()) {
        const i = r.qubits.indexOf(q);
        if (i >= 0) lines.push(`measure q[${q}] -> ${r.name}[${i}];`);
      }
      continue;
    }

    const cc = g.classicalControl;
    const ccBits = cc ? (cc.qubits ?? [cc.qubit]) : [];
    const prefix = !cc ? '' : ccBits.length === 1 ? `if(c${ccBits[0]}==${cc.equals}) ` : `if(${regs.get(ccBits.join(',')).name}==${cc.equals}) `;

    if (g.type === 'RX' || g.type === 'RY' || g.type === 'RZ') {
      const op = g.type.toLowerCase();
      const theta = g.params?.theta ?? 0;
      lines.push(`${prefix}${op}(${theta}) q[${targets[0]}];`);
      continue;
    }

    const op = QASM_OP[g.type];
    if (!op) {
      lines.push(`// skipped unsupported gate: ${g.type}`);
      continue;
    }
    const qubits = [...controls, ...targets].map((q) => `q[${q}]`).join(',');
    lines.push(`${prefix}${op} ${qubits};`);
  }

  return lines.join('\n') + '\n';
}

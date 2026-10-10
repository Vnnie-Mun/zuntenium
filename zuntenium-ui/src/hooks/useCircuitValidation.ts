// src/hooks/useCircuitValidation.ts
// ── Prompt 4: Live circuit validation matching Zuntenium backend rules ─────
import { useMemo } from "react";
import type { UICircuit, ValidationResult, ValidationIssue } from "@/types";
import { GATE_META } from "@/lib/gates";

// Backend limits
const SV_MAX_QUBITS  = 28;
const MPS_MAX_QUBITS = 200;
const PARAM_RANGE    = { min: -2 * Math.PI * 2, max: 4 * Math.PI };

export function validateCircuit(
  circuit:  UICircuit,
  backend?: string,
): ValidationResult {
  const issues: ValidationIssue[] = [];
  const { numQubits, gates } = circuit;

  // ── 1. Qubit count checks ─────────────────────────────────────────────────
  if (numQubits < 1) {
    issues.push({ severity:"error", message:"Circuit must have at least 1 qubit." });
  }
  if (numQubits > MPS_MAX_QUBITS) {
    issues.push({ severity:"error", message:`Maximum 200 qubits supported. Got ${numQubits}.` });
  }
  if (numQubits > SV_MAX_QUBITS && (!backend || backend === "statevector")) {
    issues.push({
      severity:"warning",
      message:`Statevector backend supports ≤${SV_MAX_QUBITS} qubits. Switch to TensorNetwork (MPS).`,
    });
  }

  // ── 2. Gate-level checks ──────────────────────────────────────────────────
  // Track occupancy: Map<qubit, Set<col>>
  const occupied = new Map<number, Set<number>>();

  for (const gate of gates) {
    const meta = GATE_META[gate.name];
    if (!meta) {
      issues.push({ severity:"error", gateId: gate.id, message:`Unknown gate "${gate.name}".` });
      continue;
    }

    // Qubit bounds
    const allQubits = [gate.qubit];
    if (gate.target  !== undefined) allQubits.push(gate.target);
    if (gate.target2 !== undefined) allQubits.push(gate.target2);

    for (const q of allQubits) {
      if (q < 0 || q >= numQubits) {
        issues.push({
          severity:"error", gateId: gate.id,
          message:`${meta.fullName} on qubit ${q} — out of range (circuit has ${numQubits} qubits).`,
        });
      }
    }

    // Duplicate qubits in same gate
    if (new Set(allQubits).size !== allQubits.length) {
      issues.push({
        severity:"error", gateId: gate.id,
        message:`${meta.fullName} cannot target the same qubit twice.`,
      });
    }

    // 2-qubit gates need a distinct target
    if (meta.isTwoQubit && gate.target === undefined) {
      issues.push({
        severity:"error", gateId: gate.id,
        message:`${meta.fullName} requires a target qubit.`,
      });
    }

    // 3-qubit gates
    if (meta.isThreeQubit && (gate.target === undefined || gate.target2 === undefined)) {
      issues.push({
        severity:"error", gateId: gate.id,
        message:`${meta.fullName} (Toffoli) requires control1, control2, and target.`,
      });
    }

    // Parameter range
    if (meta.hasParams) {
      for (let i = 0; i < gate.params.length; i++) {
        const p = gate.params[i] ?? 0;
        if (!isFinite(p)) {
          issues.push({
            severity:"error", gateId: gate.id,
            message:`${meta.fullName} parameter ${i+1} is not a valid number.`,
          });
        } else if (p < PARAM_RANGE.min || p > PARAM_RANGE.max) {
          issues.push({
            severity:"warning", gateId: gate.id,
            message:`${meta.fullName} θ=${p.toFixed(3)} is outside typical range [−4π, 4π].`,
          });
        }
      }
    }

    // Overlap check (same qubit, same column)
    for (const q of allQubits) {
      if (!occupied.has(q)) occupied.set(q, new Set());
      const cols = occupied.get(q)!;
      if (cols.has(gate.col)) {
        issues.push({
          severity:"error", gateId: gate.id,
          message:`Overlap: qubit Q${q} col ${gate.col} already occupied.`,
        });
      }
      cols.add(gate.col);
    }
  }

  // ── 3. Empty circuit ──────────────────────────────────────────────────────
  if (gates.length === 0) {
    issues.push({ severity:"info", message:"Circuit is empty. Add gates to run." });
  }

  return {
    valid: issues.every(i => i.severity !== "error"),
    issues,
  };
}

// Hook for use in React components
export function useCircuitValidation(circuit: UICircuit, backend?: string): ValidationResult {
  return useMemo(
    () => validateCircuit(circuit, backend),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(circuit), backend]
  );
}

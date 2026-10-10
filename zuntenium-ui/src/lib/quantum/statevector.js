import { baseMatrix, CONTROLLED_BASE } from "./gates.js";
import { applyNoiseToQubit } from "./noise.js";

export const MAX_QUBITS = 20; // 2^20 complex amplitudes ≈ 16MB of doubles; safe upper bound for a browser tool

// A statevector quantum simulator using bit-level amplitude pairing rather
// than building full 2^n x 2^n matrices — this is the standard approach for
// exact simulation and is what makes 15-20 qubit circuits tractable at all.
//
// Qubit indexing convention: bit `q` of a basis index `i` (i.e. (i >> q) & 1)
// is the value of qubit q. Basis labels are printed with qubit (n-1) as the
// leftmost character and qubit 0 as the rightmost, matching how the bits
// naturally read in `i.toString(2)`.
export class StateVector {
  constructor(numQubits, random = Math.random) {
    this.random = random;
    if (
      !Number.isInteger(numQubits) ||
      numQubits < 1 ||
      numQubits > MAX_QUBITS
    ) {
      throw new Error(
        `numQubits must be an integer between 1 and ${MAX_QUBITS}`,
      );
    }
    this.n = numQubits;
    this.size = 1 << numQubits;
    this.re = new Float64Array(this.size);
    this.im = new Float64Array(this.size);
    this.re[0] = 1; // |00...0>
  }

  // Apply a 2x2 matrix to `target`, only where every bit in `controlMask` is 1.
  _applyMatrix(target, matrix, controlMask = 0) {
    const bit = 1 << target;
    const [[m00, m01], [m10, m11]] = matrix;
    for (let i = 0; i < this.size; i++) {
      if ((i & bit) !== 0) continue; // only visit the "0" half of each pair
      if ((i & controlMask) !== controlMask) continue;
      const j = i | bit;
      const a0re = this.re[i],
        a0im = this.im[i];
      const a1re = this.re[j],
        a1im = this.im[j];
      this.re[i] =
        m00.re * a0re - m00.im * a0im + m01.re * a1re - m01.im * a1im;
      this.im[i] =
        m00.re * a0im + m00.im * a0re + m01.re * a1im + m01.im * a1re;
      this.re[j] =
        m10.re * a0re - m10.im * a0im + m11.re * a1re - m11.im * a1im;
      this.im[j] =
        m10.re * a0im + m10.im * a0re + m11.re * a1im + m11.im * a1re;
    }
  }

  applySwap(a, b) {
    if (a === b) return;
    const ba = 1 << a,
      bb = 1 << b;
    for (let i = 0; i < this.size; i++) {
      const bitA = i & ba ? 1 : 0;
      const bitB = i & bb ? 1 : 0;
      if (bitA === 0 && bitB === 1) {
        const j = i - bb + ba; // clear bit b, set bit a -> the paired basis state
        const tre = this.re[i],
          tim = this.im[i];
        this.re[i] = this.re[j];
        this.im[i] = this.im[j];
        this.re[j] = tre;
        this.im[j] = tim;
      }
    }
  }

  applyGate(gate) {
    const { type, controls = [], targets = [], params = {} } = gate;

    if (type === "MEASURE") return; // handled by measureAndCollapse() in runTrajectory(), not here
    if (type === "SWAP") {
      if (targets.length !== 2)
        throw new Error("SWAP needs exactly 2 target qubits");
      return this.applySwap(targets[0], targets[1]);
    }

    const baseType = CONTROLLED_BASE[type] || type;
    const matrix = baseMatrix(baseType, params);
    const target = targets[0];
    if (target === undefined)
      throw new Error(`Gate ${type} is missing a target qubit`);

    const controlMask = controls.reduce((mask, q) => mask | (1 << q), 0);
    this._applyMatrix(target, matrix, controlMask);
  }

  run(gates) {
    // Gates must be applied in circuit order (by column) for correctness
    // when operations don't commute. Classically-controlled gates are
    // skipped here — their condition depends on a real mid-circuit
    // measurement outcome, which doesn't exist in this coherent, no-collapse
    // evolution; runTrajectory() is where those are actually evaluated.
    const ordered = [...gates].sort(
      (a, b) => (a.column ?? 0) - (b.column ?? 0),
    );
    for (const gate of ordered) {
      if (gate.classicalControl) continue;
      this.applyGate(gate);
    }
  }

  // Reinitializes to |00...0> in place, reusing the existing typed arrays.
  // Lets a trajectory loop run thousands of shots against ONE allocation
  // instead of allocating two fresh Float64Arrays per shot.
  reset() {
    this.re.fill(0);
    this.im.fill(0);
    this.re[0] = 1;
  }

  clone() {
    const c = Object.create(StateVector.prototype);
    c.n = this.n;
    c.random = this.random;
    c.size = this.size;
    c.re = this.re.slice();
    c.im = this.im.slice();
    return c;
  }

  // <psi|Pauli_q|psi> computed by applying the Pauli to a scratch copy and
  // taking the inner product with the original — reuses the same
  // _applyMatrix path already checked against known Bell/GHZ results, so
  // there's no separate (and separately-buggy) formula to get wrong.
  expectation(target, pauliType) {
    const scratch = this.clone();
    scratch._applyMatrix(target, baseMatrix(pauliType));
    let re = 0;
    for (let i = 0; i < this.size; i++) {
      const ar = this.re[i],
        ai = this.im[i];
      const br = scratch.re[i],
        bi = scratch.im[i];
      re += ar * br + ai * bi;
    }
    return re; // Hermitian observable: imaginary part is ~0 by construction
  }

  blochVector(target) {
    // Single O(size) pass over conjugate-amplitude pairs, no clone/matrix-
    // apply needed — replaces what used to be three full-array clones (one
    // per Pauli) with one pass computing all three components at once.
    // Verified against the old expectation()-based implementation (kept
    // below for that cross-check and because it's the clearer reference
    // derivation) before this replaced it.
    const bit = 1 << target;
    let z = 0,
      xRe = 0,
      yIm = 0;
    for (let i = 0; i < this.size; i++) {
      if (i & bit) continue;
      const j = i | bit;
      const a0re = this.re[i],
        a0im = this.im[i];
      const a1re = this.re[j],
        a1im = this.im[j];
      z += a0re * a0re + a0im * a0im - (a1re * a1re + a1im * a1im);
      // conj(a0) * a1 = (a0re - i·a0im)(a1re + i·a1im)
      xRe += a0re * a1re + a0im * a1im;
      yIm += a0re * a1im - a0im * a1re;
    }
    return { qubit: target, x: round(2 * xRe), y: round(2 * yIm), z: round(z) };
  }

  blochVectors() {
    return Array.from({ length: this.n }, (_, q) => this.blochVector(q));
  }

  // Applies gates in column order, then stochastically applies a noise
  // channel to every qubit a gate touched — a standard quantum-trajectory
  // (Monte Carlo wavefunction) unraveling of a noise channel. Averaging
  // many trajectories' single-shot outcomes reproduces the noisy
  // measurement statistics a density-matrix simulation would give, without
  // needing a full O(4^n) density matrix.
  // Amplitude damping (T1-style decay toward |0>), via the standard
  // quantum-jump / Monte Carlo wavefunction unraveling of the two Kraus
  // operators E0 = |0><0| + sqrt(1-gamma)|1><1|, E1 = sqrt(gamma)|0><1|.
  // `gamma` here is used directly as this step's jump probability (a
  // simplification for an interactive tool — see the comment in noise.js —
  // not a literal continuous-time T1 rate), but the two branches applied are
  // the exact Kraus operators, correctly renormalized either way.
  applyAmplitudeDamping(target, gamma) {
    const bit = 1 << target;
    let p1 = 0;
    for (let i = 0; i < this.size; i++) {
      if (i & bit) p1 += this.re[i] * this.re[i] + this.im[i] * this.im[i];
    }
    if (p1 < 1e-12) return; // nothing populates |1> on this qubit — no-op

    if (this.random() < gamma * p1) {
      // Jump: the |1> branch is what survives, remapped onto |0>; whatever
      // was in the |0> branch is annihilated (E1's action, not a sum).
      const norm = Math.sqrt(p1);
      for (let i = 0; i < this.size; i++) {
        if (i & bit) {
          const j = i & ~bit;
          this.re[j] = this.re[i] / norm;
          this.im[j] = this.im[i] / norm;
          this.re[i] = 0;
          this.im[i] = 0;
        }
      }
    } else {
      // No-jump: shrink the |1> branch, renormalize the whole vector.
      const scale = Math.sqrt(Math.max(0, 1 - gamma));
      let normSq = 0;
      for (let i = 0; i < this.size; i++) {
        if (i & bit) {
          this.re[i] *= scale;
          this.im[i] *= scale;
        }
        normSq += this.re[i] * this.re[i] + this.im[i] * this.im[i];
      }
      const norm = Math.sqrt(normSq);
      for (let i = 0; i < this.size; i++) {
        this.re[i] /= norm;
        this.im[i] /= norm;
      }
    }
  }

  // Projective measurement: samples an outcome from this qubit's current
  // marginal, collapses the statevector onto it, and renormalizes — the
  // real physics of measuring mid-circuit, not a no-op placeholder.
  measureAndCollapse(target) {
    const bit = 1 << target;
    let p1 = 0;
    for (let i = 0; i < this.size; i++) {
      if (i & bit) p1 += this.re[i] * this.re[i] + this.im[i] * this.im[i];
    }
    const outcome = this.random() < p1 ? 1 : 0;
    const norm = Math.sqrt(outcome === 1 ? p1 : Math.max(1e-15, 1 - p1));
    for (let i = 0; i < this.size; i++) {
      const bitVal = i & bit ? 1 : 0;
      if (bitVal === outcome) {
        this.re[i] /= norm;
        this.im[i] /= norm;
      } else {
        this.re[i] = 0;
        this.im[i] = 0;
      }
    }
    return outcome;
  }

  // Runs one full trajectory: applies gates in column order, collapsing the
  // state for real on every MEASURE (so anything that happens afterward —
  // to that qubit or, via entanglement, any other — sees the post-collapse
  // state, not the pre-measurement superposition), skipping any gate whose
  // `classicalControl` condition isn't met by a measurement taken so far
  // (real classical control flow, not just collapse), and optionally
  // injecting noise after each gate that actually fires. Ends with a fresh
  // full measurement of every qubit, so the returned label is this
  // trajectory's actual final outcome.
  runTrajectory(gates, noise) {
    const ordered = [...gates].sort(
      (a, b) => (a.column ?? 0) - (b.column ?? 0),
    );
    const classicalBits = {};
    for (const gate of ordered) {
      if (gate.type === "MEASURE") {
        classicalBits[gate.targets[0]] = this.measureAndCollapse(
          gate.targets[0],
        );
        continue;
      }
      if (gate.classicalControl) {
        // Register condition: bits listed LSB first; single-bit form uses `qubit`.
        const { qubit, qubits, equals } = gate.classicalControl;
        const regBits = qubits ?? [qubit];
        if (regBits.some((q) => classicalBits[q] === undefined)) continue; // bit not yet measured — gate doesn't fire
        const value = regBits.reduce(
          (acc, q, i) => acc + (classicalBits[q] << i),
          0,
        );
        if (value !== equals) continue;
      }
      this.applyGate(gate);
      if (noise && noise.model && noise.model !== "none") {
        const qubits = [
          ...new Set([...(gate.controls || []), ...(gate.targets || [])]),
        ];
        for (const q of qubits)
          applyNoiseToQubit(this, q, noise.model, noise.p ?? 0.02, this.random);
      }
    }
    const bits = new Array(this.n);
    for (let q = 0; q < this.n; q++) bits[q] = this.measureAndCollapse(q);
    return bits.slice().reverse().join("");
  }

  probabilities() {
    const p = new Float64Array(this.size);
    for (let i = 0; i < this.size; i++)
      p[i] = this.re[i] * this.re[i] + this.im[i] * this.im[i];
    return p;
  }

  label(i) {
    return i.toString(2).padStart(this.n, "0");
  }

  // Returns the full (or, for large state spaces, top-K by probability)
  // statevector as plain JSON-serializable entries. Accepts a precomputed
  // `probs` array (from a prior probabilities() call) to avoid recomputing
  // the O(size) pass when the caller also needs it for sample() — as
  // routes/simulate.js does for every non-trajectory run.
  toStatevectorJSON({ maxEntries = 256, probs } = {}) {
    const p = probs || this.probabilities();
    const entries = [];
    for (let i = 0; i < this.size; i++) {
      if (p[i] > 1e-10 || this.size <= 64) {
        entries.push({
          index: i,
          label: this.label(i),
          amplitude: { re: round(this.re[i]), im: round(this.im[i]) },
          probability: round(p[i]),
          phase: round(Math.atan2(this.im[i], this.re[i])),
        });
      }
    }
    entries.sort((a, b) => b.probability - a.probability);
    const truncated = entries.length > maxEntries;
    return {
      entries: entries.slice(0, maxEntries),
      truncated,
      totalNonzero: entries.length,
    };
  }

  // Draws `shots` samples from the exact Born-rule distribution — this is
  // real sampling from the computed probabilities, not a canned/randomized
  // placeholder. Accepts a precomputed `probs` array for the same reason as
  // toStatevectorJSON above.
  sample(shots, probs) {
    const p = probs || this.probabilities();
    const cumulative = new Float64Array(this.size);
    let acc = 0;
    for (let i = 0; i < this.size; i++) {
      acc += p[i];
      cumulative[i] = acc;
    }
    const counts = {};
    for (let s = 0; s < shots; s++) {
      const r = this.random() * acc;
      let lo = 0,
        hi = this.size - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cumulative[mid] < r) lo = mid + 1;
        else hi = mid;
      }
      const label = this.label(lo);
      counts[label] = (counts[label] || 0) + 1;
    }
    return counts;
  }
}

function round(x) {
  return Math.abs(x) < 1e-12 ? 0 : Math.round(x * 1e6) / 1e6;
}

// True iff some qubit has an operation AFTER a MEASURE on that same qubit.
// This is exactly the condition under which "ignore MEASURE, sample the
// final joint statevector" stops being equivalent to the real physics — the
// deferred-measurement principle guarantees they're equivalent otherwise,
// so this also doubles as "do we actually need per-shot trajectories".
export function hasMidCircuitMeasurement(gates) {
  const lastTouch = {}; // qubit -> max column touched by ANY gate
  const measureColumn = {}; // qubit -> column of its MEASURE, if any
  for (const g of gates) {
    const qubits = [...(g.controls || []), ...(g.targets || [])];
    for (const q of qubits) {
      lastTouch[q] = Math.max(lastTouch[q] ?? -Infinity, g.column ?? 0);
      if (g.type === "MEASURE") {
        measureColumn[q] = Math.max(
          measureColumn[q] ?? -Infinity,
          g.column ?? 0,
        );
      }
    }
  }
  return Object.keys(measureColumn).some(
    (q) => lastTouch[q] > measureColumn[q],
  );
}

export function hasClassicalControl(gates) {
  return gates.some((g) => g.classicalControl);
}

# ZQL-11 | Error Mitigation & Correction

## Learning Objectives
- Understand the difference between mitigation and correction.
- Apply Zero Noise Extrapolation (ZNE) and Probabilistic Error Cancellation (PEC) using Zuntenium.

---

## 1. Mitigation vs. Correction

**Quantum Error Correction (QEC)** encodes a single logical qubit into many physical qubits to actively detect and correct errors (e.g., the Surface Code). This requires thousands of qubits and is decades away.

**Quantum Error Mitigation (QEM)** is a near-term strategy. It runs the circuit multiple times with varying noise profiles or sampled operations, then uses classical post-processing to infer the ideal (noiseless) result.

## 2. Using the Zuntenium Mitigation Engine

Zuntenium provides a centralized `ErrorMitigationEngine` that automatically selects the best strategy (ZNE, PEC, or Readout Inversion) based on the provided noise model.

```python
from zuntenium.error_mitigation.mitigation import ErrorMitigationEngine
from zuntenium.error_mitigation.noise_model import NoiseModel, DepolarizingNoise

# 1. Assume we have a noisy execution (e.g. from real hardware)
# Here are some mock raw counts with heavy noise
raw_counts = {'00': 400, '01': 100, '10': 100, '11': 424}

# 2. Define or load the hardware noise model
# (If using real hardware, this is fetched automatically via MHOS)
noise = NoiseModel(depolarizing=DepolarizingNoise(p=0.05)) # 5% error

# 3. Initialize the Engine
engine = ErrorMitigationEngine(
    strategy="pec", # Or "zne", "readout", "auto"
    noise_model=noise
)

# 4. Apply mitigation
result = engine.apply(counts=raw_counts, n_qubits=2)

print("Original Counts:", raw_counts)
print("Mitigated Counts:", result.corrected_counts)
print("Fidelity Improvement:", result.fidelity_improvement)
```

## Next Steps
Now that you can mitigate errors on noisy systems, it's time to route your circuits to physical QPUs using [ZQL-12 | Real Hardware Integration](ZQL-12-real-hardware.md).

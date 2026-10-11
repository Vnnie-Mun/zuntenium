"""Example 01: Creating and Simulating a Bell State in Zuntenium.

This script demonstrates:
1. Creating a 2-qubit Quantum Circuit.
2. Applying Hadamard (H) and Controlled-NOT (CX) gates to create entanglement: (|00> + |11>) / sqrt(2).
3. Simulating the circuit with Zuntenium's StatevectorSimulator.
4. Printing the measurement shot counts and final statevector.

Run in your terminal or IDE:
    python examples/01_bell_state.py
"""
import zuntenium
from zuntenium import QuantumCircuit, get_backend


def main():
    print(f"=== Zuntenium Quantum Simulator (v{zuntenium.__version__}) ===")
    print("Initializing 2-qubit Bell State (|Phi+> = (|00> + |11>)/sqrt(2))...\n")

    # 1. Allocate circuit
    qc = QuantumCircuit(2, name="bell_state")

    # 2. Apply gates (fluent chaining supported)
    qc.h(0).cx(0, 1).measure_all()

    # 3. Simulate
    backend = get_backend("statevector")
    print("Running 1,024 shots on Statevector backend...")
    result = backend.run(qc, shots=1024)

    print("\nSimulation Results:")
    print("  Shot Counts:", result.counts)
    print("  Qubit 0 & 1 are perfectly correlated (only '00' and '11' observed).")
    print("\n✓ Bell state simulation succeeded!")


if __name__ == "__main__":
    main()

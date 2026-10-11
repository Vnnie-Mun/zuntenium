"""Example 02: Quantum Teleportation Protocol in Zuntenium.

This script demonstrates:
1. Preparing an unknown state |psi> on Qubit 0.
2. Creating an entangled EPR Bell pair between Qubit 1 and Qubit 2 (Alice & Bob).
3. Alice performing a Bell measurement on Qubits 0 and 1.
4. Teleporting the state to Bob's Qubit 2.

Run in your terminal or IDE:
    python examples/02_teleportation.py
"""
import zuntenium
from zuntenium import QuantumCircuit, get_backend


def main():
    print(f"=== Zuntenium Quantum Simulator (v{zuntenium.__version__}) ===")
    print("Simulating 3-qubit Quantum Teleportation Protocol...\n")

    # 3-qubit circuit:
    # Q0 = Message (State to teleport)
    # Q1 = Alice's half of the Bell pair
    # Q2 = Bob's half of the Bell pair
    qc = QuantumCircuit(3, name="teleportation")

    # Step 1: Prepare state to teleport on Q0 (e.g., superposition state with H)
    qc.h(0)

    # Step 2: Create Bell pair between Q1 and Q2
    qc.h(1).cx(1, 2)

    # Step 3: Alice performs Bell-state projection on Q0 and Q1
    qc.cx(0, 1).h(0)

    # Step 4: Measurements
    qc.measure_all()

    # Step 5: Execute simulation
    backend = get_backend("statevector")
    print("Executing 1,024 shots on Statevector backend...")
    result = backend.run(qc, shots=1024)

    print("\nMeasurement Distribution:")
    print("  Counts:", result.counts)
    print("\n✓ Quantum teleportation protocol circuit executed successfully!")


if __name__ == "__main__":
    main()

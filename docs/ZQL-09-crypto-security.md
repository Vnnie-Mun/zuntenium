# ZQL-09 | Cryptography and Security

## Overview
Quantum computing poses a significant threat to classical cryptographic systems like RSA and ECC, which rely on the difficulty of prime factorization and discrete logarithms. Shor's Algorithm (covered conceptually in [ZQL-04](ZQL-04-core-algorithms.md)) can break these in polynomial time.

## Post-Quantum Cryptography (PQC)

While Zuntenium is a framework for *quantum* execution, protecting classical systems requires Post-Quantum Cryptography (PQC). 

*   **Lattice-based Cryptography:** Systems like CRYSTALS-Kyber and Dilithium are currently being standardized by NIST.
*   **Hash-based Signatures:** Rely on the collision resistance of cryptographic hash functions (e.g., SPHINCS+).

## Quantum Key Distribution (QKD)

Instead of relying on mathematical complexity, QKD relies on the fundamental laws of quantum mechanics (like the No-Cloning Theorem) to distribute symmetric encryption keys securely. 

### BB84 Protocol
BB84 is the first quantum cryptography protocol. It encodes classical bits into non-orthogonal quantum states (e.g., the $|0\rangle, |1\rangle$ basis and the $|+\rangle, |-\rangle$ basis). If an eavesdropper intercepts the qubits, they collapse the state, creating detectable errors.

*Status:* **Planned (Q4 2026)**
* The `zuntenium.crypto` module will offer simulated networked environments to test QKD protocols like BB84 and E91 before deploying to physical optical networks.

import numpy as np
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.hybrid_api import run_vqe, run_qaoa
from zuntenium.hybrid.observable import Hamiltonian, PauliString

def test_docs_zql05_vqe_and_qaoa():
    # VQE test
    H_vqe = Hamiltonian([(0.5, PauliString("ZZ")), (-0.3, PauliString("XX"))])
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.0, 0)
    qc.ry(0.0, 1)

    initial_params = np.array([0.1, 0.2])
    result_vqe = run_vqe(
        ansatz=qc,
        hamiltonian=H_vqe,
        initial_params=initial_params,
        max_iter=50,
        shots=0
    )
    assert result_vqe.energy is not None

    # QAOA test
    H_qaoa = Hamiltonian([(1.0, PauliString("ZZ"))])
    result_qaoa = run_qaoa(
        hamiltonian=H_qaoa,
        p=1,
        shots=0,
        max_iter=50
    )
    assert result_qaoa.energy is not None

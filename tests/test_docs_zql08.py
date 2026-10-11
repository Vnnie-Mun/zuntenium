from zuntenium.api.hybrid_api import run_qaoa
from zuntenium.hybrid.observable import Hamiltonian, PauliString

def test_docs_zql08_qaoa_maxcut():
    H_cost = Hamiltonian([
        (1.0, PauliString("ZZI")),
        (1.0, PauliString("IZZ")),
        (1.0, PauliString("ZIZ"))
    ])

    result = run_qaoa(
        hamiltonian=H_cost,
        p=2,
        shots=0,
        max_iter=100
    )
    assert result.energy is not None
    assert len(result.optimal_params) == 4

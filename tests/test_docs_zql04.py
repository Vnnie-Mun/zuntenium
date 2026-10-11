import math
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute

def test_docs_zql04_qft3():
    qc = QuantumCircuit(3, name="QFT3")
    qc.h(2)
    qc.crz(math.pi / 2, 1, 2)
    qc.crz(math.pi / 4, 0, 2)
    qc.h(1)
    qc.crz(math.pi / 2, 0, 1)
    qc.h(0)
    qc.swap(0, 2)

    result = execute(qc, shots=0)
    assert len(result.statevector) == 8
    for amp in result.statevector:
        assert math.isclose(abs(amp), 1.0 / math.sqrt(8), rel_tol=1e-5)

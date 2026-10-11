from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute

def test_docs_zql03_half_adder():
    qc = QuantumCircuit(3, name="HalfAdder")
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.measure(1, 0)
    qc.measure(2, 1)

    result = execute(qc, shots=1024)
    assert result.most_probable == "101"

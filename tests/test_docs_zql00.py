from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute

def test_docs_zql00_bell_state():
    circuit = QuantumCircuit(2, name="BellState")
    circuit.h(0).cx(0, 1).measure_all()
    result = execute(circuit, shots=1024)
    assert result.most_probable in ["00", "11"]
    assert "00" in result.counts and "11" in result.counts

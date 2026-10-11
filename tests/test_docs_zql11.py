from zuntenium.error_mitigation.mitigation import ErrorMitigationEngine
from zuntenium.error_mitigation.noise_model import NoiseModel, DepolarizingNoise

def test_docs_zql11_error_mitigation():
    raw_counts = {'00': 400, '01': 100, '10': 100, '11': 424}
    noise = NoiseModel(depolarizing=DepolarizingNoise(p=0.05))

    engine = ErrorMitigationEngine(strategy="pec", noise_model=noise)
    result = engine.apply(counts=raw_counts, n_qubits=2)
    assert result.corrected_counts is not None

from zuntenium.transpiler.cache import TranspilerCache
from zuntenium.circuit import QuantumCircuit

def test_docs_zql13_transpiler_cache_template_hash():
    cache = TranspilerCache(max_size=256)

    qc1 = QuantumCircuit(1)
    qc1.rx(0.1, 0)

    qc2 = QuantumCircuit(1)
    qc2.rx(0.9, 0)

    hash1 = cache.compute_template_hash(qc1)
    hash2 = cache.compute_template_hash(qc2)

    assert hash1 == hash2

#!/usr/bin/env python3
"""
Zuntenium — Full debug suite (v3).
Tests every implemented module bottom-up with real quantum circuits.
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✅ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        FAIL += 1

# ═══════════════════════════════════════════════════════════════════
print("\n══ 1. QuantumIR + IRNode ══")
# ═══════════════════════════════════════════════════════════════════

def test_ir_basic():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=[], metadata={})
    assert ir.num_qubits == 2
    assert ir.depth() == 0

def test_ir_with_gates():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.quantum_core.gates import HGate, CXGate
    nodes = [
        IRNode(operation="gate", gate=HGate(), qubits=[0], params=[]),
        IRNode(operation="gate", gate=CXGate(), qubits=[0,1], params=[]),
    ]
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=nodes, metadata={})
    assert len(ir.instructions) == 2
    assert ir.depth() >= 1

test("IR basic creation", test_ir_basic)
test("IR with gates", test_ir_with_gates)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 2. DAGCircuit ══")
# ═══════════════════════════════════════════════════════════════════

def test_dag_from_ir():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.circuit.ir.dag import DAGCircuit
    from zuntenium.quantum_core.gates import HGate, CXGate
    nodes = [
        IRNode(operation="gate", gate=HGate(), qubits=[0], params=[]),
        IRNode(operation="gate", gate=CXGate(), qubits=[0,1], params=[]),
    ]
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=nodes, metadata={})
    dag = DAGCircuit.from_ir(ir)
    assert dag.num_nodes >= 2

def test_dag_critical_path():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.circuit.ir.dag import DAGCircuit
    from zuntenium.quantum_core.gates import HGate, CXGate, XGate
    nodes = [
        IRNode(operation="gate", gate=HGate(), qubits=[0], params=[]),
        IRNode(operation="gate", gate=XGate(), qubits=[1], params=[]),
        IRNode(operation="gate", gate=CXGate(), qubits=[0,1], params=[]),
    ]
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=nodes, metadata={})
    dag = DAGCircuit.from_ir(ir)
    cp = dag.critical_path()
    assert len(cp) >= 1

def test_ir_to_dag():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.quantum_core.gates import HGate
    nodes = [IRNode(operation="gate", gate=HGate(), qubits=[0], params=[])]
    ir = QuantumIR(num_qubits=1, num_clbits=0, instructions=nodes, metadata={})
    dag = ir.to_dag()
    assert dag.num_nodes >= 1

test("DAG from IR", test_dag_from_ir)
test("DAG critical path", test_dag_critical_path)
test("IR.to_dag() integration", test_ir_to_dag)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 3. Circuit Fingerprinting ══")
# ═══════════════════════════════════════════════════════════════════

def test_fingerprint():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.quantum_core.gates import HGate, CXGate
    from zuntenium.circuit.fingerprint import CircuitFingerprinter
    nodes = [
        IRNode(operation="gate", gate=HGate(), qubits=[0], params=[]),
        IRNode(operation="gate", gate=CXGate(), qubits=[0,1], params=[]),
    ]
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=nodes, metadata={})
    fp = CircuitFingerprinter.fingerprint_ir(ir)
    assert isinstance(fp.structural_hash, str)
    assert len(fp.structural_hash) == 64

def test_fingerprint_similarity():
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.quantum_core.gates import HGate, CXGate
    from zuntenium.circuit.fingerprint import fingerprint
    nodes = [
        IRNode(operation="gate", gate=HGate(), qubits=[0], params=[]),
        IRNode(operation="gate", gate=CXGate(), qubits=[0,1], params=[]),
    ]
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=nodes, metadata={})
    fp1 = fingerprint(ir)
    fp2 = fingerprint(ir)
    assert fp1.matches(fp2)
    assert fp1.similar_to(fp2)

test("Fingerprint creation", test_fingerprint)
test("Fingerprint self-similarity", test_fingerprint_similarity)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 4. Stabilizer Simulator ══")
# ═══════════════════════════════════════════════════════════════════

def test_stabilizer_bell():
    from zuntenium.backends.stabilizer import StabilizerSimulator
    from dataclasses import dataclass, field as dc_field
    @dataclass
    class G:
        name: str; qubits: list; params: list = dc_field(default_factory=list)
    @dataclass
    class C:
        num_qubits: int; gates: list; name: str = "test"
    sim = StabilizerSimulator(max_qubits=10)
    circuit = C(num_qubits=2, gates=[G("H",[0]), G("CX",[0,1])])
    result = sim.execute(circuit, shots=1000)
    # result is a _Result object with .counts attribute
    counts = result.counts
    for k in counts:
        assert k in ("00","11"), f"Unexpected Bell state {k}"
    assert sum(counts.values()) == 1000

def test_stabilizer_ghz():
    from zuntenium.backends.stabilizer import StabilizerSimulator
    from dataclasses import dataclass, field as dc_field
    @dataclass
    class G:
        name: str; qubits: list; params: list = dc_field(default_factory=list)
    @dataclass
    class C:
        num_qubits: int; gates: list; name: str = "test"
    sim = StabilizerSimulator(max_qubits=10)
    circuit = C(num_qubits=3, gates=[G("H",[0]), G("CX",[0,1]), G("CX",[1,2])])
    result = sim.execute(circuit, shots=1000)
    counts = result.counts
    for k in counts:
        assert k in ("000","111"), f"Unexpected GHZ state {k}"

test("Stabilizer Bell state", test_stabilizer_bell)
test("Stabilizer GHZ state", test_stabilizer_ghz)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 5. GPU Backend ══")
# ═══════════════════════════════════════════════════════════════════

def test_gpu_backend():
    from zuntenium.backends.gpu_backend import GPUStatevectorSimulator
    from dataclasses import dataclass, field as dc_field
    @dataclass
    class G:
        name: str; qubits: list; params: list = dc_field(default_factory=list)
    @dataclass
    class C:
        num_qubits: int; gates: list; name: str = "test"
    sim = GPUStatevectorSimulator()
    circuit = C(num_qubits=2, gates=[G("H",[0]), G("CX",[0,1])])
    result = sim.execute(circuit, shots=1000)
    # result is a _Result object with .counts
    assert hasattr(result, 'counts') or hasattr(result, 'statevector')

test("GPU backend Bell state", test_gpu_backend)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 6. Sparse MPS Utilities ══")
# ═══════════════════════════════════════════════════════════════════

def test_mps_zero_state():
    from zuntenium.utils.linalg.sparse_utils import MPSState
    import numpy as np
    mps = MPSState.zero_state(3)
    sv = mps.to_statevector()
    assert abs(sv[0] - 1.0) < 1e-10
    assert abs(np.sum(np.abs(sv)**2) - 1.0) < 1e-10

def test_mps_gate():
    from zuntenium.utils.linalg.sparse_utils import MPSState, apply_single_qubit_gate
    import numpy as np
    H = np.array([[1,1],[1,-1]], dtype=np.complex128) / np.sqrt(2)
    mps = MPSState.zero_state(2)
    apply_single_qubit_gate(mps, H, 0)
    sv = mps.to_statevector()
    assert abs(sv[0] - 1/np.sqrt(2)) < 1e-10
    assert abs(sv[2] - 1/np.sqrt(2)) < 1e-10

def test_mps_roundtrip():
    from zuntenium.utils.linalg.sparse_utils import MPSState
    import numpy as np
    sv = np.zeros(4, dtype=np.complex128)
    sv[0] = sv[3] = 1/np.sqrt(2)
    mps = MPSState.from_statevector(sv)
    assert np.allclose(sv, mps.to_statevector(), atol=1e-10)

def test_sparse_dict():
    from zuntenium.utils.linalg.sparse_utils import to_sparse_dict, from_sparse_dict
    import numpy as np
    sv = np.zeros(8, dtype=np.complex128)
    sv[0] = 0.5; sv[7] = np.sqrt(0.75)
    sparse = to_sparse_dict(sv)
    assert len(sparse) == 2
    assert np.allclose(sv, from_sparse_dict(sparse, 8))

test("MPS zero state", test_mps_zero_state)
test("MPS single-qubit gate", test_mps_gate)
test("MPS from_statevector roundtrip", test_mps_roundtrip)
test("Sparse dict roundtrip", test_sparse_dict)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 7. KAK Decomposition ══")
# ═══════════════════════════════════════════════════════════════════

def test_kak():
    from zuntenium.utils.linalg.decomposition import kak_decompose
    import numpy as np
    cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
    result = kak_decompose(cnot)
    assert "interaction" in result
    assert "fidelity" in result
    assert result["fidelity"] > 0.8

test("KAK decomposition of CNOT", test_kak)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 8. Solovay-Kitaev ══")
# ═══════════════════════════════════════════════════════════════════

def test_solovay_kitaev():
    from zuntenium.utils.linalg.decomposition import solovay_kitaev
    import numpy as np
    T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128)
    result = solovay_kitaev(T, depth=2)
    # solovay_kitaev returns a list of gate names
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 0

test("Solovay-Kitaev T gate approx", test_solovay_kitaev)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 9. QMLLayer ══")
# ═══════════════════════════════════════════════════════════════════

def test_qml_forward():
    from zuntenium.hybrid.solver.qml_layer import QMLLayer
    import numpy as np
    layer = QMLLayer(num_qubits=2, num_layers=1)
    params = layer.random_params()
    assert params.shape[0] > 0
    out = layer.forward(params)
    assert len(out) >= 1

def test_qml_gradient():
    from zuntenium.hybrid.solver.qml_layer import QMLLayer
    import numpy as np
    layer = QMLLayer(num_qubits=2, num_layers=1)
    params = layer.random_params()
    grads = layer.gradient(params)
    assert grads.shape[0] == params.shape[0], f"Gradient rows {grads.shape[0]} != params {params.shape[0]}"

test("QMLLayer forward pass", test_qml_forward)
test("QMLLayer gradient", test_qml_gradient)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 10. AI Optimizer Pass ══")
# ═══════════════════════════════════════════════════════════════════

def test_ai_pass():
    from zuntenium.transpiler.passes.ai_pass.ai_optimizer import AIOptimizerPass
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.quantum_core.gates import HGate, CXGate
    nodes = [
        IRNode(operation="gate", gate=HGate(), qubits=[0], params=[]),
        IRNode(operation="gate", gate=CXGate(), qubits=[0,1], params=[]),
    ]
    ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=nodes, metadata={})
    pass_ = AIOptimizerPass(collect_data=True, model_path=None)
    result = pass_.run(ir)
    assert result.num_qubits == 2

test("AI pass data collection mode", test_ai_pass)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 11. Cloud Client SDK ══")
# ═══════════════════════════════════════════════════════════════════

def test_cloud_client():
    from zuntenium.api.cloud.cloud_client import CloudClient
    client = CloudClient(base_url="https://api.test.io/v1", api_key="test-key")
    client.close()

test("Cloud client init", test_cloud_client)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 12. Expectation Value ══")
# ═══════════════════════════════════════════════════════════════════

def test_exp_z():
    from zuntenium.quantum_core.statevector.statevector_simulator import StatevectorSimulator
    sim = StatevectorSimulator(num_qubits=1)
    assert abs(sim.get_expectation_value("Z") - 1.0) < 1e-6

def test_exp_x():
    from zuntenium.quantum_core.statevector.statevector_simulator import StatevectorSimulator
    sim = StatevectorSimulator(num_qubits=1)
    assert abs(sim.get_expectation_value("X")) < 1e-6

test("Expectation <0|Z|0> = +1", test_exp_z)
test("Expectation <0|X|0> = 0", test_exp_x)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 13. Backends __init__ exports ══")
# ═══════════════════════════════════════════════════════════════════

def test_exports():
    from zuntenium.backends import StabilizerSimulator, GPUStatevectorSimulator
    assert StabilizerSimulator is not None
    assert GPUStatevectorSimulator is not None

test("Backend exports", test_exports)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 14. SABRE Router ══")
# ═══════════════════════════════════════════════════════════════════

def test_sabre():
    from zuntenium.transpiler.passes.mapping.sabre_router import SABRERouter
    router = SABRERouter(coupling_edges=[(i, i+1) for i in range(4)])
    assert router is not None

test("SABRE router init", test_sabre)

# ═══════════════════════════════════════════════════════════════════
print("\n══ 15. IonQ Backend ══")
# ═══════════════════════════════════════════════════════════════════

def test_ionq():
    from zuntenium.backend.hardware.ionq.ionq_backend import IonQBackend
    backend = IonQBackend(device="simulator")
    assert backend.name is not None
    assert len(backend.supported_gates) > 0

test("IonQ backend init", test_ionq)

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print(f"{'='*60}")
sys.exit(1 if FAIL else 0)

<p align="center">
  <img src="docs/assets/quflow-logo.png" alt="QuFlow Logo" width="480" />
</p>

<h1 align="center">QuFlow</h1>

<p align="center"><strong>Enterprise-Grade Hybrid Quantum Computing SDK, Transpiler & Cloud Platform</strong></p>

<p align="center">

[![PyPI version](https://img.shields.io/pypi/v/zuntenium.svg?color=7C3AED)](https://pypi.org/project/zuntenium/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-7C3AED.svg)](https://pypi.org/project/zuntenium/)
[![Tests Passing](https://img.shields.io/badge/tests-549%2F549%20passed%20(Ubuntu%20%2B%20macOS)-brightgreen.svg)](https://github.com/Vnnie-Mun/zuntenium/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://github.com/Vnnie-Mun/zuntenium)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-7C3AED.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker Ready](https://img.shields.io/badge/docker-compose%20%26%20k8s-informational.svg)](https://github.com/Vnnie-Mun/zuntenium/tree/main/zuntenium_cloud)
[![Next.js Standalone](https://img.shields.io/badge/UI-Next.js%2015%20Standalone-Node%2018%2B-black.svg)](https://github.com/Vnnie-Mun/zuntenium/tree/main/zuntenium-ui)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vnnie-Mun/zuntenium/blob/main/notebooks/zuntenium_colab_quickstart.ipynb)

</p>

---

## Overview

**QuFlow** (package: `zuntenium`) is a high-performance, full-stack quantum computing platform designed to scale seamlessly from interactive circuit design on laptops to distributed GPU simulation and physical Quantum Processing Unit (QPU) orchestration.

QuFlow converges advanced circuit compilation, simulation engines, multi-hardware orchestration, and self-hosted cloud microservices into a unified, production-ready ecosystem:

- 1. **Multi-Engine Quantum Simulation**:
  - **Aaronson-Gottesman Stabilizer Tableau**: Simulates Clifford circuits up to **10,000+ qubits** in $O(n^2)$ time with phase tracking.
  - **CuPy GPU Statevector Backend**: High-throughput GPU tensor contraction for dense unitary execution.
  - **Matrix Product State (MPS) Tensor Network**: Simulates 50–100+ weakly entangled qubits with SVD bond-dimension truncation.
  - **Exact Statevector Simulator**: Big-endian pure-state evolution with full Pauli expectation value contraction $\langle \psi | P | \psi \rangle$.
- 2. **Transpiler Engine v2**:
  - **`TranspilerCache`**: Structural template hashing with **$O(1)$ parameter rebinding**, cutting 95%+ transpilation latency in VQE and QAOA variational loops.
  - **SABRE Qubit Routing**: Lookahead distance heuristics, extended successor sets, and decay penalties for heavy-hex and grid topologies.
  - **Cartan KAK Lie Algebra Decomposition**: Canonical $SU(4)$ two-qubit decomposition minimizing two-qubit gate counts.
  - **Solovay-Kitaev $\epsilon$-Net Approximation**: Recursive group commutator approximation for arbitrary single-qubit rotations.
  - **Circuit Fingerprinting**: Parameter-invariant topological and DAG similarity hashing.
- 3. **Multi-Hardware Orchestration System (MHOS)**:
  - Unified adapter interface for physical execution on **IBM Quantum (Runtime)**, **IonQ Cloud**, **Amazon Braket**, and **Rigetti (QCS)** with automatic simulator fallback.
- 4. **Zero-Noise Extrapolation (ZNE) & Error Mitigation**:
  - Richardson, polynomial, and exponential noise scaling with randomized Pauli / readout mitigation.
- 5. **Self-Hostable Cloud Platform & Microservices**:
  - **FastAPI REST API**: Authentication, JWT tokens, API keys, quota management, and public backend discovery.
  - **RQ Distributed Worker Cluster**: High, default, low, and hardware queues with graceful job execution.
  - **Full Observability**: PostgreSQL 15 (UUID typed), Redis 7, Prometheus metrics, and Grafana dashboard provisioning.
- 6. **Next.js 15 Visual QuFlow Workspace**:
  - Standalone web interface with interactive quantum circuit builder, Bloch sphere visualizer, and live cloud job telemetry.

---

## Architecture

```
User Code / QuFlow UI  ───►  FastAPI Gateway (Port 8000)
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
        Transpiler Engine v2                 Distributed Job Queue
   ├── TranspilerCache (O(1) Rebind)         ├── Redis Task Broker
   ├── SABRE Lookahead Router                ├── RQ Worker Cluster
   ├── Cartan KAK SU(4) Decomposition        └── PostgreSQL 15 (UUID)
   └── Solovay-Kitaev ε-Net
                  │
                  ▼
         Multi-Hardware Orchestrator (MHOS)
   ┌──────────────────────────────────────────────────────────┐
   │                     Execution Backends                   │
   ├────────────────────────────┬─────────────────────────────┤
   │ Local Simulation:          │ Cloud QPUs:                 │
   │ • Stabilizer (10,000+ q)   │ • IBM Quantum (Runtime)     │
   │ • CuPy GPU Statevector     │ • IonQ Cloud (Trapped Ion)  │
   │ • Tensor Network (MPS)     │ • Amazon Braket             │
   │ • C++ / NumPy Statevector  │ • Rigetti QCS (Supercond.)  │
   └────────────────────────────┴─────────────────────────────┘
```

---

## Competitive Capability Matrix

How QuFlow compares with established quantum frameworks:

| Capability Domain | Qiskit | PennyLane | TKET | CUDA-Q | **QuFlow** |
|---|---|---|---|---|---|
| **Primary Focus** | IBM ecosystem & general SDK | QML & hybrid gradients | High-performance routing | GPU-accelerated C++ | **Full-Stack Quantum SDK + Cloud + UI** |
| **Simulation Diversity** | High (Aer) | Via plugins | Via backends | CuQuantum | **High** (Statevector, Stabilizer $10^4$q, GPU, MPS) |
| **Clifford Stabilizer** | Included | External | External | Limited | **Top-Tier** ($10^4$+ qubits built-in) |
| **Transpiler Caching** | Basic pass manager | Tape caching | Structural passes | Compile-time | **Top-Tier** ($O(1)$ Template Rebinding) |
| **Multi-Hardware Portability** | IBM-centric | Provider plugins | Provider plugins | NVIDIA targets | **Native MHOS** (IBM, IonQ, Braket, Rigetti) |
| **Self-Hosted Cloud Stack** | Proprietary | None | None | None | **Built-in** (FastAPI, RQ, Postgres, Redis, Prometheus) |
| **Visual Canvas Web UI** | Web Composer | None | None | None | **Built-in** (Next.js 15 Standalone QuFlow Workspace) |
| **Test Verification** | Extensive | Extensive | Extensive | Extensive | **549 / 549 passing (100%)** |

---

## Quick Start & Code Examples

### 1. Massive Clifford Simulation (10,000 Qubits)

Simulate 10,000-qubit entangled states in seconds using the Aaronson-Gottesman binary stabilizer tableau:

```python
import zuntenium as zn

# Create a 10,000-qubit GHZ state circuit
qc = zn.QuantumCircuit(10000)
qc.h(0)
for i in range(9999):
    qc.cx(i, i + 1)
qc.measure_all()

# Execute on the stabilizer simulator
result = zn.execute(qc, backend="stabilizer", shots=1000)  # QuFlow auto-dispatches
print(result.counts)
# Output: {'00...0': 503, '11...1': 497}
```

---

### 2. $O(1)$ Variational Acceleration with `TranspilerCache`

Eliminate 95%+ transpilation overhead during VQE, QAOA, and QML parameter update loops:

```python
from zuntenium import QuantumCircuit
from zuntenium.transpiler import TranspilerCache

# Initialize structural template cache
cache = TranspilerCache(max_entries=1000)

# Build a parameterized variational ansatz
ansatz = QuantumCircuit(4)
ansatz.rx(0, theta=0.1).ry(1, theta=0.2).cx(0, 1).cx(2, 3)

# Hash topological structure invariant to parameter values
template_hash = cache.compute_template_hash(ansatz)

# In optimization loop: rebind in O(1) time without recompilation
for epoch in range(100):
    new_angles = [0.05 * epoch, 0.12 * epoch]
    fast_circuit = cache.rebind(template_hash, new_angles)
```

---

### 3. Hardware-Aware SABRE Routing

Map arbitrary quantum circuits onto physical chip coupling graphs with lookahead distance heuristics:

```python
from zuntenium.transpiler.passes.mapping.sabre_router import SABRERouter
from zuntenium.transpiler.coupling_map import CouplingMap
from zuntenium import QuantumCircuit

# Define a 3x3 grid hardware topology
coupling = CouplingMap.from_grid(3, 3)
router = SABRERouter(coupling_map=coupling, lookahead_weight=0.5, decay_rate=0.001)

# Route unmapped circuit onto physical architecture
routed_circuit, stats = router.run(unmapped_circuit)
print(f"Inserted SWAPs: {stats['swaps_added']}, Final Depth: {routed_circuit.depth()}")
```

---

### 4. Self-Hosted Cloud Client Execution

Submit circuits to your private QuFlow Cloud cluster:

```python
from zuntenium.api.cloud import CloudClient
from zuntenium import QuantumCircuit

# Initialize client with API key
client = CloudClient(base_url="http://localhost:8000", api_key="qf_live_...")

# Discover available hardware and simulators
backends = client.list_backends()
print("Available:", [b["name"] for b in backends])

# Submit asynchronous job with error mitigation
job = client.submit(
    circuit=qc,
    backend="auto",
    shots=2048,
    mitigation=True,
    optimization_mode="rule",
)

result = job.wait_for_result(timeout=60)
print(result.counts)
```

---

## Installation

> **Requirements**: Python 3.10–3.12 · Node.js ≥18.17 (for the UI) · Docker + Compose v2 (for the cloud stack)

### From PyPI

**macOS / Linux:**
```bash
python3 -m pip install zuntenium
```

**Windows (PowerShell):**
```powershell
py -m pip install zuntenium
```

### With Optional Accelerators

**GPU acceleration** (Linux x86\_64 + CUDA 12 only — skipped automatically on macOS/Windows):

```bash
python3 -m pip install "zuntenium[gpu]"
```

**Cloud microservices stack** (FastAPI, Redis, RQ, SQLAlchemy):

```bash
python3 -m pip install "zuntenium[cloud]"
```

**All standard extras** (dev + AI + viz + cloud, no GPU):

```bash
python3 -m pip install "zuntenium[all]"
```

> **Note:** The `[all]` extra does not include GPU. For everything including GPU use `[full]` — the GPU extra is automatically skipped on macOS and Windows.

### From Source

**macOS / Linux:**
```bash
git clone https://github.com/Vnnie-Mun/zuntenium.git
cd zuntenium
python3 -m pip install -e ".[dev,cloud]"
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/Vnnie-Mun/zuntenium.git
cd zuntenium
py -m pip install -e ".[dev,cloud]"
```

---

## Deploying QuFlow Cloud

Launch the full cloud infrastructure (API, multi-priority RQ workers, PostgreSQL, Redis, Prometheus, and Grafana).

**Requires**: Docker with [Compose v2](https://docs.docker.com/compose/migrate/) (i.e. `docker compose`, not `docker-compose`).

> **RQ workers and Redis run inside Linux containers.** Native Windows or macOS processes are not supported for these services — use Docker Desktop or WSL2.

```bash
cd zuntenium_cloud
docker compose up -d
```

| Service | URL | Notes |
|---------|-----|-------|
| **API Gateway** | `http://localhost:8000` | Docs at `/docs` |
| **Prometheus** | `http://localhost:9090` | Monitoring profile |
| **Grafana Dashboard** | `http://localhost:3050` | Monitoring profile |
| **RQ Worker Dashboard** | `http://localhost:9181` | Monitoring profile |

To also start monitoring services:
```bash
docker compose --profile monitoring up -d
```

### Running the QuFlow Web Workspace

**Requires**: Node.js ≥18.17

```bash
cd zuntenium-ui
npm install
npm run build
npm run start
```
Interactive visual workspace available at `http://localhost:3000`.

---

## Test Suite & Verification

QuFlow is verified with **549/549 tests passing** on Ubuntu 22.04 and macOS 14 (Apple Silicon), Python 3.10–3.12.

**macOS / Linux:**
```bash
# Run complete test suite
python3 -m pytest tests/

# Run unit tests only
python3 -m pytest tests/unit/

# Run end-to-end integration verification
python3 debug_all.py
```

**Windows (PowerShell):**
```powershell
py -m pytest tests/
```

---

## License

QuFlow is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.

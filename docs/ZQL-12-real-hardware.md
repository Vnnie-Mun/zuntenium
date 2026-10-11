# ZQL-12 | Real Hardware Integration

## Learning Objectives
- Route a quantum circuit to real QPU hardware (IBM, Rigetti, IonQ).
- Understand Zuntenium's Multi-Hardware Orchestration System (MHOS).
- Utilize FAILOVER and RACING execution strategies.

---

## 1. Multi-Hardware Orchestration System (MHOS)

Zuntenium does not lock you into a single hardware vendor. The MHOS intelligently routes your circuit to the best available QPU based on queue times, cost, and historical fidelity (tracked by the internal AI-Engine).

Supported Providers:
*   IBM Quantum (Superconducting)
*   IonQ (Trapped Ion)
*   Rigetti (Superconducting)
*   AWS Braket (Multi-vendor)

## 2. Using HardwareOrchestrator

You can submit your circuits directly to the MHOS orchestrator. By default, it uses the `FAILOVER` strategy: if the best device goes offline during queueing, it automatically falls back to the next best device, and finally to the MPS Simulator.

```python
import asyncio
from zuntenium.circuit import QuantumCircuit
from zuntenium.mhos.orchestrator import HardwareOrchestrator, ExecutionStrategy
from zuntenium.mhos.hardware_registry import HardwareRegistry
from zuntenium.mhos.hardware_selector import HardwareSelector

# 1. Create your circuit
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1).measure_all()

# 2. Setup MHOS Infrastructure (Requires valid Provider API Keys in .env)
registry = HardwareRegistry()
selector = HardwareSelector(registry)
mhos = HardwareOrchestrator(registry, selector)

# 3. Execute with RACING strategy
# Submits to the top 3 devices simultaneously and returns the first completed result.
async def run_on_hardware():
    result = await mhos.execute(
        circuit=qc,
        shots=1024,
        strategy=ExecutionStrategy.RACING
    )
    print(f"Device Used: {result.device_used} ({result.provider_used})")
    print("Counts:", result.counts)

# asyncio.run(run_on_hardware())
```

## Next Steps
Now that you can run on real hardware, learn how to cache your expensive transpilation jobs and contribute to Zuntenium's research engine in [ZQL-13 | Research Track](ZQL-13-research-track.md).

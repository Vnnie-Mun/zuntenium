# REALITY MAP (F1–F12): Developer Experience & IDE Workability

This audit maps the Zuntenium SDK against the 12 Developer Experience, Static Analysis, Packaging, and Notebook Realities (F1–F12).

---

## Reality Matrix Summary

| ID | Reality Name | Current SDK Behavior | Evidence Citation | Gap Rating |
|:---|:---|:---|:---|:---:|
| **F1** | Static Analysis Realities | No `py.typed` marker; `zuntenium/__init__.py` has untyped dynamic `__getattr__`; language servers show unknown types | [`zuntenium/__init__.py:34`](file:///home/vincent/Downloads/zuntenium/zuntenium/__init__.py#L34), missing `py.typed` | **weak** |
| **F2** | Dynamic Python Defeats Analysis | 10 primary classes hidden behind runtime `_LAZY_EXPORTS` dict; no `if TYPE_CHECKING:` stubs | [`zuntenium/__init__.py:20-31`](file:///home/vincent/Downloads/zuntenium/zuntenium/__init__.py#L20) | **weak** |
| **F3** | Invisible Compiled Modules | `_zuntenium_core.so` lives in `zuntenium/`, but stub is misnamed `_zuntenium_cpp.pyi` inside a subfolder | [`zuntenium/bindings/cpp_bindings/_zuntenium_cpp.pyi`](file:///home/vincent/Downloads/zuntenium/zuntenium/bindings/cpp_bindings/_zuntenium_cpp.pyi) | **weak** |
| **F4** | Interpreter & Env Mismatches | `zuntenium info` exists but lacks diagnostic `doctor` command for environment, PATH, and IDE interpreter checks | [`zuntenium/cli.py:26-46`](file:///home/vincent/Downloads/zuntenium/zuntenium/cli.py#L26) | **weak** |
| **F5** | Binary Compatibility Limits | Prebuilt `.so` is Linux x86_64 CPython 3.12 only; no cibuildwheel matrix for macOS/Windows/Python 3.10/3.11 | [`zuntenium/_zuntenium_core.cpython-312-x86_64-linux-gnu.so`](file:///home/vincent/Downloads/zuntenium/zuntenium/_zuntenium_core.cpython-312-x86_64-linux-gnu.so) | **weak** |
| **F6** | Dependency Conflicts | `pip check` fails (`pynacl` missing `cffi`); optional dependencies grouped in `pyproject.toml` | Terminal output: `pip check` exit 1; [`pyproject.toml:50-103`](file:///home/vincent/Downloads/zuntenium/pyproject.toml#L50) | **weak** |
| **F7** | Notebook Rendering Protocols | `QuantumCircuit._repr_html_` hardcodes dark mode (`#0d1117`); `ExecutionResult` lacks `_repr_html_` and static SVG fallback | [`circuit.py:348-356`](file:///home/vincent/Downloads/zuntenium/zuntenium/circuit/circuit.py#L348), [`result.py:168`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/executor/result.py#L168) | **weak** |
| **F8** | Kernel Event Loops & Async | Direct calls to `asyncio.get_event_loop()` without detecting already running notebook loops | [`zuntenium/mhos/connectors/ibm.py:65`](file:///home/vincent/Downloads/zuntenium/zuntenium/mhos/connectors/ibm.py#L65) | **weak** |
| **F9** | Native Kernel Crashes | Boundary input validation exists in Python; C++ statevector allocates without checking OS available RAM | [`circuit.py:65`](file:///home/vincent/Downloads/zuntenium/zuntenium/circuit/circuit.py#L65), [`statevector.cpp:16-24`](file:///home/vincent/Downloads/zuntenium/bindings_core/cpp/src/statevector.cpp#L16) | **weak** |
| **F10**| Debugger & Inspector Safety | `ExecutionResult.__repr__` summarizes top 3 counts; `QuantumCircuit` and `ExecutionResult` are picklable | Empirical pickle test verified; [`result.py:168-174`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/executor/result.py#L168) | **good** |
| **F11**| Import Cost & LS Latency | Cold import is 221 ms; lazy import dict minimizes cold time, but hurts static analysis | Benchmark: 221.01 ms; [`zuntenium/__init__.py:20`](file:///home/vincent/Downloads/zuntenium/zuntenium/__init__.py#L20) | **good** |
| **F12**| Example Correctness | Both Python example scripts (`01_bell_state.py`, `02_teleportation.py`) and quickstart notebook pass headlessly | Empirical test run: 100% pass | **good** |

---

## Detailed Analysis & Evidence

### F1 & F2: Static Analysis Realities & Dynamic Python
- **Fact**: Language servers (Pylance, Pyright, Jedi, PyCharm) do not execute code. They construct an abstract syntax tree and resolve type definitions statically.
- **Evidence**:
  - `zuntenium/` contains NO `py.typed` file. Type checkers default to treating the library as untyped.
  - In [`zuntenium/__init__.py:20-31`](file:///home/vincent/Downloads/zuntenium/zuntenium/__init__.py#L20):
    ```python
    _LAZY_EXPORTS: dict[str, tuple[str, str]] = {
        "HardwareAwareTranspiler": ("zuntenium.transpiler", "HardwareAwareTranspiler"),
        "IBMBackend": ("zuntenium.backends", "IBMBackend"),
        "AdaptiveQuantumExecutor": ("zuntenium.execution", "AdaptiveQuantumExecutor"),
        ...
    }
    def __getattr__(name: str) -> Any:
        ...
    ```
    Typing `import zuntenium as z; z.IBMBackend` causes language servers to show red squiggles (`"IBMBackend" is unknown attribute`) or provide zero autocomplete and hover signatures.
- **Solution Lever**: Add `py.typed` to `zuntenium/` and use `from typing import TYPE_CHECKING` blocks in `__init__.py` to declare explicit static imports for all public symbols without incurring cold-import runtime overhead.

### F3: Compiled Extension Stubs
- **Fact**: Binary extensions (`.so`, `.pyd`) contain compiled machine code that language servers cannot parse. Without `.pyi` stubs co-located with the module, IDEs report missing module or unknown symbols.
- **Evidence**:
  - The compiled extension is named `_zuntenium_core.cpython-312-x86_64-linux-gnu.so` and sits in `zuntenium/`.
  - The only `.pyi` file in the repo is located at `zuntenium/bindings/cpp_bindings/_zuntenium_cpp.pyi`.
  - Because neither the name (`_zuntenium_cpp.pyi` vs `_zuntenium_core.so`) nor the directory matches, Pylance/Pyright cannot locate stubs for `zuntenium._zuntenium_core`.
- **Solution Lever**: Generate and ship `zuntenium/_zuntenium_core.pyi` directly alongside the binary in the package root.

### F4: Interpreter Mismatch & Diagnostics
- **Fact**: The #1 support ticket across IDEs (VS Code, Cursor, PyCharm, Neovim) is interpreter mismatch (user runs terminal in one venv, IDE runs system Python).
- **Evidence**:
  - `zuntenium cli.py` provides `version`, `info`, `backends`, `run`.
  - There is no `zuntenium doctor` or `zqc doctor` command that validates active `sys.executable`, package paths, wheel architectures, C++ extensions, and suggests exact IDE interpreter configuration steps.
- **Solution Lever**: Implement `zuntenium doctor` and provide automated diagnostic JSON + rich terminal output.

### F5 & F6: Binary Compatibility & Dependency Conflicts
- **Fact**: Shipping a C++ `.so` built for Linux x86_64 CPython 3.12 means Windows, macOS, and Linux Python 3.10/3.11 fail to load native acceleration, and tight dependency pins break co-installed scientific packages.
- **Evidence**:
  - Running `pip check` currently flags `pynacl 1.5.0 requires cffi, which is not installed`.
  - The compiled `.so` only matches CPython 3.12 on Linux x86_64.
- **Solution Lever**: Setup GitHub Actions `cibuildwheel` matrix for Python 3.10–3.12 across Windows, macOS (ARM64/x86_64), and Linux manylinux. Clean up dependency pins and test co-installation with Qiskit, Cirq, PyTorch, and SciPy.

### F7: Notebook Display Protocols
- **Fact**: Notebooks (JupyterLab, Colab, VS Code) render objects using `_repr_html_`, `_repr_mimebundle_`, and `__repr__`. Hardcoded CSS creates unreadable output in light themes.
- **Evidence**:
  - In [`zuntenium/circuit/circuit.py:348-356`](file:///home/vincent/Downloads/zuntenium/zuntenium/circuit/circuit.py#L348):
    ```python
    def _repr_html_(self) -> str:
        text = self.draw()
        return (
            f"<div style='background-color: #0d1117; color: #58a6ff; ...'>"
            f"{text}</div>"
        )
    ```
    This renders as an illegible dark rectangle on standard white Jupyter/Colab notebooks.
  - `ExecutionResult` lacks `_repr_html_`, so running `execute(qc)` in a notebook prints basic text `ExecutionResult(...)` instead of an interactive HTML result table or SVG bar chart.
- **Solution Lever**: Implement theme-adaptive CSS (`var(--jp-content-font-color)`, CSS `@media (prefers-color-scheme: dark)`) with clean light/dark contrast, and provide rich HTML rendering for `ExecutionResult`.

### F8: Async Kernel Collisions
- **Fact**: Jupyter kernels run an active `asyncio` event loop. Calling `asyncio.get_event_loop().run_until_complete()` throws `RuntimeError: This event loop is already running`.
- **Evidence**:
  - [`zuntenium/mhos/connectors/ibm.py:65`](file:///home/vincent/Downloads/zuntenium/zuntenium/mhos/connectors/ibm.py#L65) calls `loop = asyncio.get_event_loop()` without checking if the loop is currently running.
- **Solution Lever**: Detect running event loops using `asyncio.get_running_loop()`. Provide seamless synchronous APIs using background worker threads or `nest_asyncio` when inside a running loop.

### F9: Native Safety & Memory Checks
- **Fact**: An unhandled native crash kills the Jupyter kernel or IDE debugger with zero Python traceback.
- **Evidence**:
  - `StateVectorCPP` in C++ allocates `uint64_t(1) << num_qubits` elements directly without validating available physical memory.
- **Solution Lever**: Add pre-flight memory validation before calling the C++ allocator, raising clean Python `MemoryError` before segmentation faults or OS OOM kills occur.

### F10, F11, F12: Debugging, Import Cost & Examples
- **Evidence**:
  - `ExecutionResult` repr is size-safe (only dumps top 3 counts).
  - Core objects are picklable.
  - Cold import time is 221 ms.
  - Examples and quickstart notebook pass verification cleanly.
- **Gap Rating**: **good**.

# Zuntenium Quantum IDE & Environment Support Matrix

This matrix tracks the developer experience, static analysis, debugging, and rendering fidelity of the Zuntenium SDK and libraries across target environments.

Status Legend:
- **VERIFIED**: Empirically tested and confirmed working with zero errors, accurate output, and proper completions.
- **FAILING**: Tested and produces static-analysis false squiggles, missing completions, broken rendering, or runtime failures.
- **UNTESTED**: Pending automated/manual environment certification in Phase 3.

---

## Environment Certification Matrix

| IDE / Environment | Version | Install | Interpreter Detection | Autocomplete | Hover Docs | Type Checking | Debugging | Notebook Rendering | Async Jobs | GPU Support | Output Correctness |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **VS Code (Python + Pylance)** | 1.80+ / 1.94 | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | VERIFIED (F10) | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **PyCharm (Community / Pro)** | 2024.2+ | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | VERIFIED (F10) | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **Google Colab** | Python 3.10 | VERIFIED | VERIFIED | FAILING (F2/F3) | FAILING (F2) | UNTESTED | UNTESTED | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **JupyterLab** | 4.2+ | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | UNTESTED | VERIFIED (F10) | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **Jupyter Notebook 7** | 7.x | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | UNTESTED | UNTESTED | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **Cursor** | 0.42+ | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | VERIFIED (F10) | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **Antigravity IDE** | Current | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | VERIFIED (F10) | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **Neovim (LSP / Pyright)** | 0.10+ | VERIFIED | FAILING (F4) | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | UNTESTED | UNTESTED | FAILING (F8) | UNTESTED | VERIFIED |
| **Spyder** | 6.0+ | VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | UNTESTED | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **GitHub Codespaces** | Ubuntu 22.04| VERIFIED | VERIFIED | FAILING (F1/F2) | FAILING (F2) | FAILING (F1) | VERIFIED (F10) | FAILING (F7) | FAILING (F8) | UNTESTED | VERIFIED |
| **QCanvas Web Visual IDE** | Next.js 15 | VERIFIED | N/A (Web UI) | VERIFIED (Palette)| VERIFIED | VERIFIED | N/A | VERIFIED | VERIFIED | UNTESTED | VERIFIED |

---

## Failure Root Causes (Mapped to Realities F1–F12)

1. **Autocomplete & Hover Docs (FAILING in all Python language servers)**:
   - Root Cause **F1**: `py.typed` (PEP 561) marker is completely missing from `zuntenium/`. Language servers run in non-typed fallback mode.
   - Root Cause **F2**: 10 major public symbols (`IBMBackend`, `HardwareAwareTranspiler`, `AdaptiveQuantumExecutor`, `TensorNetworkBackend`, etc.) are defined through a runtime `__getattr__` dynamic lookup dictionary in `zuntenium/__init__.py`. Static analysis engines (Pylance, Pyright, Jedi) cannot statically infer these attributes without `if TYPE_CHECKING:` explicit imports.
   - Root Cause **F3**: `_zuntenium_core.so` lacks a co-located `_zuntenium_core.pyi` stub file.

2. **Notebook Rendering (FAILING in Light Theme / Dark Mode Clashes)**:
   - Root Cause **F7**: `QuantumCircuit._repr_html_` in `zuntenium/circuit/circuit.py` uses hardcoded CSS (`background-color: #0d1117; color: #58a6ff;`). It renders an illegible dark block in light mode Jupyter/Colab themes.
   - Root Cause **F7**: `ExecutionResult` and `Statevector` lack `_repr_html_` and static SVG render fallbacks.

3. **Async Jobs in Kernels (FAILING in Jupyter/IPython)**:
   - Root Cause **F8**: `zuntenium/mhos/connectors/ibm.py` calls `loop = asyncio.get_event_loop()` and runs blocking waits on active event loops, raising `RuntimeError: This event loop is already running` in IPython/Jupyter notebook environments.

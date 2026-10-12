# Zuntenium SDK & Workspace: Certification Tiers

This document defines the certification levels, qualification criteria, and test cadence for all supported IDEs, editors, notebook environments, and operating systems.

---

## 1. Certification Tier Definitions

### Tier 1: Certified (Primary Support)
- **Definition**: Fully tested and verified across all 5 core scenarios (Fresh Install, Authoring/Autocomplete, Execution & Output, Debugging, Hardware/Cloud Path).
- **Quality SLA**:
  - 100% automated regression test coverage in CI on every release candidate.
  - Zero static analysis errors or broken completions in default configurations.
  - Maximum 48-hour Mean Time to Resolution (MTTR) for compatibility regressions.
- **Environments in Tier 1**:
  - **VS Code** (v1.80+) with Python & Pylance extensions
  - **PyCharm Professional / Community** (2024.2+)
  - **JupyterLab** (v4.2+) & Jupyter Notebook 7
  - **Google Colab** (Standard Linux Python 3.10 runtime)
  - **QCanvas Visual Quantum IDE** (Next.js 15)

---

### Tier 2: Supported (Secondary Support)
- **Definition**: Verified for standard execution, typing, and notebook workflows. Advanced features (mixed-language C++ debugging, custom visual extensions) may require manual setup documented in our guides.
- **Quality SLA**:
  - Automated smoke and authoring tests run weekly and prior to minor releases.
  - Documented setup instructions with verified configuration snippets.
  - Maximum 7-day MTTR for compatibility issues.
- **Environments in Tier 2**:
  - **Cursor** (v0.40+)
  - **Antigravity IDE**
  - **GitHub Codespaces** (Ubuntu 22.04 LTS container)
  - **Spyder** (v6.0+)
  - **Neovim** (v0.10+) with `nvim-lspconfig` and `pyright`

---

### Tier 3: Experimental (Community & Emerging)
- **Definition**: Functional via standard Python packaging and OpenQASM APIs, but not continuously tested against new IDE releases in our automated farm.
- **Quality SLA**:
  - Best-effort community fixes.
  - User-reported issues tracked against F-numbers.
- **Environments in Tier 3**:
  - **Visual Studio 2022** / **CLion** (C++ native extension developers)
  - **Emacs** with `lsp-mode`
  - **Sublime Text 4** with LSP
  - **Kaggle Notebooks**

---

## 2. Five-Scenario Certification Gates

Every Tier 1 environment must pass these 5 scenarios before a release is certified:

1. **Scenario 1: Fresh Installation & Cold Start**
   - Clean virtual environment or container.
   - `pip install zuntenium` completes without build failures.
   - `zqc doctor` passes and detects the correct interpreter.
   - Time to first circuit execution $< 5$ minutes.

2. **Scenario 2: Code Authoring & Static Analysis**
   - 100% signature accuracy and hover documentation for all public symbols (`QuantumCircuit`, `execute`, `IBMBackend`, etc.).
   - Zero false error squiggles in Pyright strict mode.
   - Parameter completion and inline type hints verified.

3. **Scenario 3: Run & Output Rendering**
   - ASCII, Unicode, and rich HTML circuit drawings display without clipping.
   - Output legibility verified in both Light Theme and Dark Theme.
   - Histogram rendering works headlessly or falls back cleanly.

4. **Scenario 4: Interactive Debugging**
   - Breakpoints hit accurately.
   - Variable inspection shows compact, size-safe representations (preventing debugger freezing on large statevectors).
   - Core objects pickle and inspect without side-effects.

5. **Scenario 5: Hardware & Async Path**
   - Asynchronous job submission functions in active notebook event loops.
   - Credentials load via environment variable or OS keyring without token leakage.

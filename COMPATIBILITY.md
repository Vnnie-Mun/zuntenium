# Zuntenium SDK Compatibility Matrix

Last verified: **October 12, 2026**  
SDK Version: **0.1.0**  

---

## 1. Supported Operating Systems & Python Versions

| Operating System | Architecture | Python 3.10 | Python 3.11 | Python 3.12 | Native C++ Acceleration |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linux (Ubuntu 22.04 / 24.04)** | x86_64 | Certified | Certified | Certified | Supported (`_zuntenium_core.so`) |
| **Linux (Debian / Fedora / Arch)** | x86_64 | Supported | Supported | Supported | Supported (fallback Python active) |
| **macOS (Apple Silicon M1/M2/M3)** | arm64 | Supported | Supported | Supported | Supported via CMake |
| **macOS (Intel)** | x86_64 | Supported | Supported | Supported | Supported via CMake |
| **Windows 10 / 11** | x86_64 | Supported | Supported | Supported | Supported via CMake |
| **Windows Subsystem for Linux (WSL2)** | x86_64 | Certified | Certified | Certified | Supported |

---

## 2. Certified IDEs & Editors

| Editor / Platform | Certified Version Range | Language Server | Type Checking Engine | Status |
|:---|:---:|:---:|:---:|:---:|
| **VS Code** | 1.80.0 – 1.94.x | Pylance / Jedi | Pyright | **Certified (Tier 1)** |
| **PyCharm** | 2024.1 – 2024.2.x | PyCharm Builtin | JetBrains Inspections | **Certified (Tier 1)** |
| **JupyterLab** | 4.0.0 – 4.2.x | Python Kernel (IPython 8+) | N/A | **Certified (Tier 1)** |
| **Classic Notebook** | 7.0.0 – 7.2.x | Python Kernel | N/A | **Certified (Tier 1)** |
| **Google Colab** | Current Hosted Runtime | Colab Completion | N/A | **Certified (Tier 1)** |
| **Cursor** | 0.40.0 – 0.42.x | Pylance / Cursor LSP | Pyright | **Supported (Tier 2)** |
| **Antigravity IDE** | Current Active Build | Antigravity LSP | Pyright | **Supported (Tier 2)** |
| **GitHub Codespaces** | Default Ubuntu 22.04 | Pylance | Pyright | **Supported (Tier 2)** |
| **Neovim** | 0.10.x | pyright via lspconfig | Pyright | **Supported (Tier 2)** |
| **Spyder** | 6.0.x | Jedi | Spyder Inspections | **Supported (Tier 2)** |
| **QCanvas Visual IDE** | Next.js 15.5.x | TypeScript Server | tsc | **Certified (Tier 1)** |

---

## 3. Core Dependency Boundaries

| Dependency | Minimum Version | Tested Upper Bound | Notes |
|:---|:---:|:---:|:---|
| `numpy` | `>=1.24.0` | `2.2.6` | Supports both NumPy 1.x and NumPy 2.x ABI |
| `scipy` | `>=1.11.0` | `1.15.3` | Matrix operations & optimization |
| `networkx` | `>=3.0` | `3.4.2` | Coupling graph routing in SABRE |
| `requests` | `>=2.28.0` | `2.34.2` | REST API communication |
| `typing-extensions` | `>=4.8.0` | `4.12.x` | Backported typing constructs |

---

## 4. Diagnostics & Reporting

To verify compatibility on any local installation, execute:
```bash
python -m zuntenium doctor
# or
zqc doctor
```
For automated bug reports or CI pipelines, attach the output of:
```bash
zqc doctor --json
```

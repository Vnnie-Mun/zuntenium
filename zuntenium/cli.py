"""
cli.py — ZUNTENIUM command-line interface.

Entry point registered as ``zuntenium`` console script.

Usage:
    zuntenium --help
    zuntenium run circuit.qasm --backend statevector --shots 1024
    zuntenium info
    zuntenium version
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Safe Unicode: fall back to ASCII on terminals that can't render box-drawing chars
_uenc = sys.stdout.encoding or "ascii"
_OK  = "[OK]"  if not _uenc.lower().startswith("utf") else "\u2713"
_ERR = "[X]"   if not _uenc.lower().startswith("utf") else "\u2717"
_BOX = "=" * 48 if not _uenc.lower().startswith("utf") else "\u2550" * 48


def cmd_version(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Print the installed ZUNTENIUM version."""
    from zuntenium.version import __version__
    print(f"ZUNTENIUM v{__version__}")


def cmd_info(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Print system info: Python version, numpy, scipy, and C++ binding status."""
    import platform

    import numpy as np
    import scipy

    from zuntenium.bindings.binding_loader import is_cpp_available
    from zuntenium.version import __version__

    cpp_status = f"{_OK} compiled" if is_cpp_available() else f"{_ERR} not built (NumPy fallback active)"
    print(_BOX)
    print(" ZUNTENIUM System Info")
    print(_BOX)
    print(f"  ZUNTENIUM version : {__version__}")
    print(f"  Python            : {platform.python_version()}")
    print(f"  NumPy             : {np.__version__}")
    print(f"  SciPy             : {scipy.__version__}")
    print(f"  C++ extension     : {cpp_status}")
    print(_BOX)


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a QASM circuit file and print measurement results."""
    from zuntenium.api.quantum_api import execute, get_backend
    from zuntenium.circuit.ir.serialization.qasm_parser import parse_qasm2

    qasm_path = Path(args.circuit)
    if not qasm_path.exists():
        print(f"Error: File not found: {qasm_path}", file=sys.stderr)
        sys.exit(1)

    qasm_str = qasm_path.read_text(encoding="utf-8")
    ir = parse_qasm2(qasm_str)

    backend = get_backend(args.backend)
    result = execute(ir, backend=backend, shots=args.shots)

    print(f"Backend   : {args.backend}")
    print(f"Shots     : {args.shots}")
    print(f"Qubits    : {result.num_qubits}")
    print(f"Most probable: |{result.most_probable}⟩")
    print("\nCounts:")
    for outcome, count in sorted(result.counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(30 * count / args.shots)
        print(f"  |{outcome}⟩: {count:6d}  {bar}")

    if args.output:
        out = {
            "counts": result.counts,
            "shots": result.shots,
            "num_qubits": result.num_qubits,
            "most_probable": result.most_probable,
            "metadata": result.metadata,
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {args.output}")


def cmd_backends(args: argparse.Namespace) -> None:  # noqa: ARG001
    """List all registered backends."""
    from zuntenium.api.quantum_api import available_backends
    print("Available backends:")
    for name in available_backends():
        print(f"  • {name}")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run full environment, IDE interpreter, native binary, and GPU diagnostics."""
    import os
    import platform
    import sys
    from pathlib import Path

    import zuntenium
    from zuntenium.bindings.binding_loader import is_cpp_available
    from zuntenium.version import __version__

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    pkg_dir = Path(zuntenium.__file__).parent
    py_typed_exists = (pkg_dir / "py.typed").exists()
    stubs_exist = (pkg_dir / "_zuntenium_core.pyi").exists()
    cpp_ok = is_cpp_available()

    # GPU Check
    gpu_found = False
    gpu_info = "None detected (CPU fallback active)"
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        gpu_found = True
        gpu_info = f"CuPy (CUDA {cp.cuda.runtime.runtimeGetVersion()})"
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_found = True
                gpu_info = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass

    # Core dependencies
    deps: dict[str, str] = {}
    for mod_name in ["numpy", "scipy", "networkx", "requests"]:
        try:
            mod = __import__(mod_name)
            deps[mod_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[mod_name] = "MISSING"

    # IDE detection
    detected_ide = "Unknown / Terminal"
    if "VSCODE_PID" in os.environ or "VSCODE_GIT_ASKPASS_NODE" in os.environ:
        detected_ide = "VS Code / Cursor"
    elif "PYCHARM_HOSTED" in os.environ:
        detected_ide = "PyCharm"
    elif "JPY_PARENT_PID" in os.environ or "COLAB_GPU" in os.environ:
        detected_ide = "Jupyter / Colab"

    report = {
        "version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "is_virtualenv": in_venv,
            "conda_env": conda_env or None,
        },
        "package": {
            "location": str(pkg_dir),
            "py_typed": py_typed_exists,
            "cpp_stubs": stubs_exist,
        },
        "native_acceleration": {
            "cpp_available": cpp_ok,
            "gpu_detected": gpu_found,
            "gpu_details": gpu_info,
        },
        "dependencies": deps,
        "environment": {
            "detected_ide": detected_ide,
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        },
    }

    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
        return

    print("═" * 60)
    print("  ZUNTENIUM Environment & IDE Health Check (zqc doctor)")
    print("═" * 60)
    print(f"  SDK Version        : v{__version__}")
    print(f"  Python Interpreter : {sys.executable}")
    print(f"  Python Version     : {platform.python_version()} ({platform.machine()})")
    print(f"  Virtual Environment: {'✓ Active' if in_venv else '✗ System Python (recommend venv)'}")
    if conda_env:
        print(f"  Conda Environment  : {conda_env}")
    print(f"  Package Location   : {pkg_dir}")
    print(f"  Type Marker        : {'✓ py.typed present' if py_typed_exists else '✗ py.typed missing'}")
    print(f"  Native Type Stubs  : {'✓ _zuntenium_core.pyi present' if stubs_exist else '✗ stubs missing'}")
    print(f"  C++ Acceleration   : {_OK + ' Active (_zuntenium_core)' if cpp_ok else _ERR + ' Pure Python fallback'}")
    print(f"  GPU Acceleration   : {_OK + ' ' + gpu_info if gpu_found else '- ' + gpu_info}")
    print(f"  Active IDE Context : {detected_ide}")
    print("\nCore Dependencies:")
    for dep, ver in deps.items():
        status = _OK if ver != "MISSING" else _ERR
        print(f"  [{status}] {dep:10s} : {ver}")

    print("\nIDE Quick Configuration:")
    print(f"  • VS Code / Cursor : Command Palette -> 'Python: Select Interpreter'")
    print(f"                       Enter: {sys.executable}")
    print(f"  • PyCharm          : Settings -> Project -> Python Interpreter -> Add Existing")
    print(f"  • Jupyter / Colab  : python -m ipykernel install --user --name zuntenium-env")
    print("═" * 60)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="zuntenium",
        description="ZUNTENIUM — AI-Native Hybrid Quantum Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zuntenium doctor
  zuntenium doctor --json
  zuntenium version
  zuntenium info
  zuntenium backends
  zuntenium run circuit.qasm --shots 2048 --backend statevector_simulator
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    doc_p = sub.add_parser("doctor", help="Run environment, IDE interpreter, and native diagnostics")
    doc_p.add_argument("--json", action="store_true", help="Output diagnostics in JSON format")

    # version
    sub.add_parser("version", help="Print ZUNTENIUM version")

    # info
    sub.add_parser("info", help="Print system info and backend status")

    # backends
    sub.add_parser("backends", help="List registered backends")

    # run
    run_p = sub.add_parser("run", help="Execute a QASM circuit file")
    run_p.add_argument("circuit", help="Path to .qasm file")
    run_p.add_argument("--backend", default="statevector_simulator",
                       help="Backend name (default: statevector_simulator)")
    run_p.add_argument("--shots", type=int, default=1024,
                       help="Number of measurement shots (default: 1024)")
    run_p.add_argument("--output", default="", help="Save results as JSON to this path")

    args = parser.parse_args()
    dispatch = {
        "doctor": cmd_doctor,
        "version": cmd_version,
        "info": cmd_info,
        "backends": cmd_backends,
        "run": cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

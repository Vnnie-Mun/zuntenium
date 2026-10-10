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
import sys
from pathlib import Path


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

    cpp_status = "✓ compiled" if is_cpp_available() else "✗ not built (NumPy fallback active)"
    print("═" * 48)
    print(" ZUNTENIUM System Info")
    print("═" * 48)
    print(f"  ZUNTENIUM version : {__version__}")
    print(f"  Python            : {platform.python_version()}")
    print(f"  NumPy             : {np.__version__}")
    print(f"  SciPy             : {scipy.__version__}")
    print(f"  C++ extension     : {cpp_status}")
    print("═" * 48)


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a QASM circuit file and print measurement results."""
    from zuntenium.api.quantum_api import execute, get_backend
    from zuntenium.circuit.ir.serialization.qasm_parser import parse_qasm2

    qasm_path = Path(args.circuit)
    if not qasm_path.exists():
        print(f"Error: File not found: {qasm_path}", file=sys.stderr)
        sys.exit(1)

    qasm_str = qasm_path.read_text()
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
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to: {args.output}")


def cmd_backends(args: argparse.Namespace) -> None:  # noqa: ARG001
    """List all registered backends."""
    from zuntenium.api.quantum_api import available_backends
    print("Available backends:")
    for name in available_backends():
        print(f"  • {name}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="zuntenium",
        description="ZUNTENIUM — AI-Native Hybrid Quantum Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zuntenium version
  zuntenium info
  zuntenium backends
  zuntenium run circuit.qasm --shots 2048 --backend statevector_simulator
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
        "version": cmd_version,
        "info": cmd_info,
        "backends": cmd_backends,
        "run": cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

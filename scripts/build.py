#!/usr/bin/env python3
"""
scripts/build.py — Cross-platform build and dev tooling for Zuntenium QuFlow.

Replaces build_bindings.sh (POSIX-only) and the POSIX-only Makefile targets.
Works identically on Windows, macOS, and Linux.

Usage:
    python scripts/build.py bindings          # Compile C++ pybind11 extension
    python scripts/build.py install           # Install package (production)
    python scripts/build.py dev-install       # Install editable + dev deps
    python scripts/build.py test             # Run full test suite
    python scripts/build.py test-unit        # Run unit tests only
    python scripts/build.py test-integration # Run integration tests only
    python scripts/build.py lint             # Run ruff linter
    python scripts/build.py format           # Auto-format code with ruff
    python scripts/build.py typecheck        # Run mypy
    python scripts/build.py clean            # Remove build artifacts
    python scripts/build.py doctor           # Print platform diagnostics

U-references: U2 (shell-neutral commands), U3 (interpreter portability),
              U5 (native-code portability), U7 (path safety).
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINDINGS_DIR = PROJECT_ROOT / "bindings"
BUILD_DIR = PROJECT_ROOT / "build" / "_cmake"
PACKAGE_DIR = PROJECT_ROOT / "zuntenium"

# Cross-platform interpreter: prefer the one running this script.
PYTHON = sys.executable

# Extension suffixes per platform (CL-17 fix: include .pyd for Windows)
_EXT_PATTERNS = ["_zuntenium_core*.so", "_zuntenium_core*.pyd", "_zuntenium_core*.dylib"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep() -> None:
    width = 50
    bar = "=" * width
    # Use Unicode box-drawing if stdout supports it
    enc = getattr(sys.stdout, "encoding", "ascii") or "ascii"
    if enc.lower().startswith("utf"):
        bar = "\u2550" * width
    print(bar)


def _run(*cmd: str, check: bool = True, cwd: Path | None = None) -> int:
    """Run a subprocess command safely (no shell=True)."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        print(f"\nERROR: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


def _find_tool(names: list[str]) -> str | None:
    """Return the first name found on PATH, or None."""
    for name in names:
        if shutil.which(name):
            return name
    return None


def _cpu_count() -> int:
    """Portable CPU count (U7: no nproc/sysctl dependency)."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _find_extension() -> Path | None:
    """Find the built extension file regardless of platform extension."""
    for pattern in _EXT_PATTERNS:
        matches = list(BUILD_DIR.rglob(pattern.replace("_zuntenium_core*", "*zuntenium_core*")))
        if matches:
            return matches[0]
    # Broader search across any name matching the patterns
    for root, _dirs, files in os.walk(BUILD_DIR):
        for fname in files:
            for pat in _EXT_PATTERNS:
                if fnmatch.fnmatch(fname, pat):
                    return Path(root) / fname
    return None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_bindings(build_type: str = "Release") -> None:
    """Compile the _zuntenium_core pybind11 C++ extension (CL-17 fix)."""
    _sep()
    print(" ZUNTENIUM — C++ Bindings Build")
    print(f" Build type : {build_type}")
    print(f" Python     : {PYTHON} ({platform.python_version()})")
    print(f" Platform   : {platform.system()} {platform.machine()}")
    _sep()

    # Check cmake
    cmake = shutil.which("cmake")
    if not cmake:
        print("WARNING: cmake not found — skipping C++ build. NumPy fallback is active.")
        print("  Install cmake: https://cmake.org/download/")
        return

    # Check C++ compiler
    if platform.system() == "Windows":
        # On Windows, MSVC is invoked by cmake; check for cl.exe or clang-cl
        cxx = _find_tool(["cl", "clang-cl", "g++", "clang++"])
        if not cxx:
            print("WARNING: No C++ compiler found (MSVC/clang-cl/g++) — skipping C++ build.")
            print("  Install Visual Studio Build Tools: https://aka.ms/vs/buildtools")
            return
    else:
        cxx = _find_tool(["g++", "clang++", "c++"])
        if not cxx:
            print("WARNING: No C++ compiler (g++/clang++) found — skipping C++ build.")
            print("  macOS: xcode-select --install")
            print("  Ubuntu/Debian: sudo apt-get install build-essential")
            print("  Fedora: sudo dnf install gcc-c++")
            return

    # Resolve pybind11 cmake dir from the running Python (U3: uses current interpreter)
    result = subprocess.run(
        [PYTHON, "-m", "pybind11", "--cmakedir"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("WARNING: pybind11 cmake dir not found — skipping C++ build.")
        print(f"  Install with: {PYTHON} -m pip install pybind11")
        return
    pybind11_dir = result.stdout.strip()
    print(f"pybind11 cmake dir: {pybind11_dir}")

    # Configure
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    print("\nConfiguring cmake...")
    _run(
        cmake,
        f"-S{BINDINGS_DIR}",
        f"-B{BUILD_DIR}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DPython_EXECUTABLE={PYTHON}",
    )

    # Build
    ncores = _cpu_count()
    print(f"\nBuilding with {ncores} parallel jobs...")
    _run(cmake, "--build", str(BUILD_DIR), "--config", build_type, "--parallel", str(ncores))

    # Install: copy extension to package directory (CL-17: handles .so, .pyd, .dylib)
    ext_file = _find_extension()
    if ext_file:
        dest = PACKAGE_DIR / ext_file.name
        shutil.copy2(ext_file, dest)
        print(f"\n[OK] Compiled : {ext_file}")
        print(f"[OK] Installed: {dest}")
    else:
        print("\nWARNING: Extension file not found in build dir — check cmake output above.")
        print(f"  Expected one of: {_EXT_PATTERNS}")

    _sep()
    print(" C++ build complete.")
    print(f" Test with: {PYTHON} -c \"import _zuntenium_core; print('C++ OK')\"")
    _sep()


def cmd_install() -> None:
    """Install package in production mode."""
    _run(PYTHON, "-m", "pip", "install", "--upgrade", "pip")
    _run(PYTHON, "-m", "pip", "install", "-r", "requirements.txt", cwd=PROJECT_ROOT)


def cmd_dev_install() -> None:
    """Install package in editable mode with dev deps."""
    _run(PYTHON, "-m", "pip", "install", "--upgrade", "pip")
    _run(PYTHON, "-m", "pip", "install", "-e", ".[dev,cloud]", cwd=PROJECT_ROOT)
    # Install pre-commit hooks if available (optional; swallowed on failure)
    pre_commit = shutil.which("pre-commit")
    if pre_commit:
        subprocess.run([pre_commit, "install"], cwd=str(PROJECT_ROOT))


def cmd_test(extra_args: list[str] | None = None) -> None:
    """Run full test suite with coverage."""
    _run(
        PYTHON, "-m", "pytest",
        "tests/",
        "--cov=zuntenium",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-v",
        *(extra_args or []),
        cwd=PROJECT_ROOT,
    )


def cmd_test_unit() -> None:
    """Run unit tests only."""
    _run(PYTHON, "-m", "pytest", "tests/unit/", "-v", cwd=PROJECT_ROOT)


def cmd_test_integration() -> None:
    """Run integration tests only."""
    _run(PYTHON, "-m", "pytest", "tests/integration/", "-v", cwd=PROJECT_ROOT)


def cmd_lint() -> None:
    """Run ruff linter."""
    _run(PYTHON, "-m", "ruff", "check", "zuntenium/", "tests/", cwd=PROJECT_ROOT)


def cmd_format() -> None:
    """Auto-format code with ruff."""
    _run(PYTHON, "-m", "ruff", "format", "zuntenium/", "tests/", cwd=PROJECT_ROOT)
    _run(PYTHON, "-m", "ruff", "check", "--fix", "zuntenium/", "tests/", cwd=PROJECT_ROOT)


def cmd_typecheck() -> None:
    """Run mypy type checker."""
    _run(PYTHON, "-m", "mypy", "zuntenium/", cwd=PROJECT_ROOT)


def cmd_clean() -> None:
    """Remove build artefacts (cross-platform; no rm -rf shell dependency)."""
    patterns = [
        "build", "dist", "*.egg-info", ".pytest_cache",
        ".mypy_cache", ".ruff_cache",
    ]
    for pat in patterns:
        for path in PROJECT_ROOT.glob(pat):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                print(f"  removed dir : {path.relative_to(PROJECT_ROOT)}")
            else:
                path.unlink(missing_ok=True)
                print(f"  removed file: {path.relative_to(PROJECT_ROOT)}")

    # Remove __pycache__ and *.pyc recursively
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # skip hidden dirs and .venv
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        for d in list(dirs):
            if d == "__pycache__":
                full = Path(root) / d
                shutil.rmtree(full, ignore_errors=True)
                print(f"  removed dir : {full.relative_to(PROJECT_ROOT)}")
                dirs.remove(d)
        for f in files:
            if f.endswith(".pyc"):
                (Path(root) / f).unlink(missing_ok=True)
            # Also remove compiled extensions from package dir
            if any(fnmatch.fnmatch(f, pat) for pat in _EXT_PATTERNS):
                full = Path(root) / f
                full.unlink(missing_ok=True)
                print(f"  removed ext : {full.relative_to(PROJECT_ROOT)}")

    print("\n[OK] Clean complete.")


def cmd_doctor() -> None:
    """Print platform diagnostics (extends zqc doctor output)."""
    _sep()
    print(" ZUNTENIUM — Build Environment Doctor")
    _sep()

    def check(label: str, value: str) -> None:
        print(f"  {label:<28}: {value}")

    check("Python", f"{sys.version}")
    check("Platform", platform.platform())
    check("Architecture", f"{platform.machine()} / {platform.architecture()[0]}")
    check("System", platform.system())

    libc = platform.libc_ver()
    check("libc", f"{libc[0]} {libc[1]}" if libc[0] else "N/A (musl or Windows)")

    check("cmake", shutil.which("cmake") or "NOT FOUND")
    for name in ["g++", "clang++", "cl"]:
        path = shutil.which(name)
        if path:
            check(f"C++ compiler ({name})", path)
            break
    else:
        check("C++ compiler", "NOT FOUND")

    # pybind11
    r = subprocess.run([PYTHON, "-m", "pybind11", "--cmakedir"],
                       capture_output=True, text=True, encoding="utf-8")
    check("pybind11 cmake dir", r.stdout.strip() if r.returncode == 0 else "NOT FOUND")

    # Docker / Compose
    docker = shutil.which("docker")
    check("docker", docker or "NOT FOUND")
    if docker:
        cr = subprocess.run([docker, "compose", "version"],
                            capture_output=True, text=True, encoding="utf-8")
        check("docker compose (v2)", cr.stdout.strip() if cr.returncode == 0 else "NOT FOUND — install Compose v2 plugin")

    # Node
    node = shutil.which("node")
    if node:
        nr = subprocess.run([node, "--version"], capture_output=True, text=True, encoding="utf-8")
        check("node", nr.stdout.strip())
    else:
        check("node", "NOT FOUND")

    npm = shutil.which("npm")
    if npm:
        npmr = subprocess.run([npm, "--version"], capture_output=True, text=True, encoding="utf-8")
        check("npm", npmr.stdout.strip())
    else:
        check("npm", "NOT FOUND")

    # GPU
    try:
        import importlib.util
        if importlib.util.find_spec("cupy"):
            check("GPU (cupy)", "available")
        else:
            check("GPU (cupy)", "not installed — CPU fallback active")
    except Exception:
        check("GPU (cupy)", "check failed")

    _sep()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/build.py",
        description="Cross-platform build and dev tooling for Zuntenium QuFlow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=[
            "bindings", "install", "dev-install",
            "test", "test-unit", "test-integration",
            "lint", "format", "typecheck",
            "clean", "doctor",
        ],
        help="Build command to run",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        choices=["Release", "Debug", "RelWithDebInfo"],
        help="CMake build type (default: Release)",
    )
    args, remainder = parser.parse_known_args()

    dispatch = {
        "bindings":         lambda: cmd_bindings(args.build_type),
        "install":          cmd_install,
        "dev-install":      cmd_dev_install,
        "test":             lambda: cmd_test(remainder),
        "test-unit":        cmd_test_unit,
        "test-integration": cmd_test_integration,
        "lint":             cmd_lint,
        "format":           cmd_format,
        "typecheck":        cmd_typecheck,
        "clean":            cmd_clean,
        "doctor":           cmd_doctor,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()

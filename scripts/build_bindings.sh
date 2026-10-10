#!/usr/bin/env bash
# scripts/build_bindings.sh
# Builds the _zuntenium_core pybind11 C++ extension module.
# Falls back gracefully if cmake/compiler/pybind11 are unavailable.
#
# Usage:
#   bash scripts/build_bindings.sh              # Release build
#   ZUNTENIUM_BUILD_TYPE=Debug bash scripts/build_bindings.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDINGS_DIR="${PROJECT_ROOT}/bindings"
BUILD_DIR="${PROJECT_ROOT}/build/_cmake"
BUILD_TYPE="${ZUNTENIUM_BUILD_TYPE:-Release}"

echo "══════════════════════════════════════════════════"
echo " ZUNTENIUM — C++ Bindings Build"
echo " CMakeLists : ${BINDINGS_DIR}/CMakeLists.txt"
echo " Build dir  : ${BUILD_DIR}"
echo " Build type : ${BUILD_TYPE}"
echo "══════════════════════════════════════════════════"

# ── Dependency checks ──────────────────────────────────────────────────────
need() {
    if ! command -v "$1" &>/dev/null; then
        echo "WARNING: '$1' not found — ${2}"
        echo "  Skipping C++ build. Python (NumPy) fallback will be used."
        exit 0
    fi
}

need cmake   "Install with: sudo apt-get install cmake"
need python3 "Install with: sudo apt-get install python3-dev"
need g++     "Install with: sudo apt-get install build-essential"

# ── Resolve Python executable (prefer venv) ────────────────────────────────
if [ -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON="$(which python3)"
fi
echo "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"

# ── Resolve pybind11 cmake dir ─────────────────────────────────────────────
# pybind11 is installed as a Python package; ask Python where its cmake
# config files are. This avoids the need for a system-wide cmake install.
PYBIND11_DIR="$(${PYTHON} -m pybind11 --cmakedir 2>/dev/null || echo "")"
if [ -z "${PYBIND11_DIR}" ]; then
    echo ""
    echo "ERROR: pybind11 cmake dir not found."
    echo "  Install with: pip install pybind11"
    echo "  Then re-run: make build-bindings"
    echo ""
    echo "  Skipping C++ build. NumPy fallback is active."
    exit 0
fi
echo "pybind11 cmake dir: ${PYBIND11_DIR}"

# ── Configure ─────────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"
echo ""
echo "Configuring cmake..."
cmake \
    -S "${BINDINGS_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    -DPython_EXECUTABLE="${PYTHON}"

# ── Build ──────────────────────────────────────────────────────────────────
NCORES="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
echo ""
echo "Building with ${NCORES} parallel jobs..."
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${NCORES}"

# ── Install: copy .so to package root ─────────────────────────────────────
SO_FILE="$(find "${BUILD_DIR}" -name "_zuntenium_core*.so" 2>/dev/null | head -1)"
if [ -n "${SO_FILE}" ]; then
    DEST="${PROJECT_ROOT}/zuntenium/"
    cp "${SO_FILE}" "${DEST}"
    echo ""
    echo "✓ Compiled: ${SO_FILE}"
    echo "✓ Installed: ${DEST}$(basename "${SO_FILE}")"
else
    echo ""
    echo "WARNING: .so not found in build dir — check cmake output above."
fi

echo ""
echo "══════════════════════════════════════════════════"
echo " C++ build complete."
echo " Test: python -c \"import _zuntenium_core; print('C++ OK')\""
echo "══════════════════════════════════════════════════"

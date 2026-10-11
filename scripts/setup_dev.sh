#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_dev.sh — One-command local development setup for VS Code & IDEs
# ==============================================================================
set -euo pipefail

echo "=== [1/4] Checking Python 3.10+ ==="
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi
python3 -c "import sys; assert sys.version_info >= (3, 10), f'Python 3.10+ required, found {sys.version}'"

echo "=== [2/4] Initializing Local Virtual Environment (.venv) ==="
# Check if existing .venv is functional or corrupted (e.g. copied/moved folder)
RECREATE_VENV=false
if [ -d ".venv" ]; then
    VENV_PY=""
    if [ -x ".venv/bin/python" ]; then
        VENV_PY=".venv/bin/python"
    elif [ -x ".venv/Scripts/python.exe" ]; then
        VENV_PY=".venv/Scripts/python.exe"
    fi
    if [ -z "$VENV_PY" ] || ! "$VENV_PY" -c "import sys" &> /dev/null; then
        echo "  ⚠ Existing .venv is broken (invalid interpreter or moved path). Recreating..."
        rm -rf .venv
        RECREATE_VENV=true
    else
        echo "  ✓ Existing working .venv found."
    fi
else
    RECREATE_VENV=true
fi

if [ "$RECREATE_VENV" = true ]; then
    if python3 -m venv .venv 2>/dev/null; then
        echo "  ✓ Created .venv via python3 -m venv."
    elif command -v virtualenv &>/dev/null && virtualenv .venv; then
        echo "  ✓ Created .venv via virtualenv."
    elif [ -x "$HOME/.local/bin/virtualenv" ] && "$HOME/.local/bin/virtualenv" .venv; then
        echo "  ✓ Created .venv via ~/.local/bin/virtualenv."
    else
        echo "  Installing virtualenv..."
        python3 -m pip install --user virtualenv || pip3 install --user virtualenv
        "$HOME/.local/bin/virtualenv" .venv
        echo "  ✓ Created .venv via user virtualenv."
    fi
fi

# Source virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

echo "=== [3/4] Installing Package Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev,cloud]"

echo "=== [4/4] Verifying Zuntenium Installation ==="
python -c "
import zuntenium
from zuntenium import QuantumCircuit, get_backend
print('  ✓ zuntenium version:', zuntenium.__version__)
qc = QuantumCircuit(2, name='bell_verification')
qc.h(0).cx(0, 1).measure_all()
backend = get_backend('statevector')
res = backend.run(qc, shots=100)
print('  ✓ Quantum Statevector Simulator passed! Counts:', res.counts)
"

echo ""
echo "===================================================================="
echo "🎉 Zuntenium development environment is ready for VS Code / your IDE!"
echo "To activate in your terminal anytime:"
echo "   source .venv/bin/activate"
echo "In VS Code:"
echo "   Press Cmd+Shift+P (or Ctrl+Shift+P) ➔ Python: Select Interpreter"
echo "   Select '.venv' (at top of list)"
echo "===================================================================="

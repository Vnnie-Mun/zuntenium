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
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  ✓ Created .venv virtual environment."
else
    echo "  ✓ Existing .venv found."
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
print('  ✓ zuntenium version:', zuntenium.__version__)
from zuntenium.core import Circuit
from zuntenium.simulators import StatevectorSimulator
circ = Circuit(2)
circ.h(0)
circ.cx(0, 1)
sim = StatevectorSimulator()
res = sim.run(circ, shots=100)
print('  ✓ Local Quantum Simulator test passed! Counts:', res.counts)
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

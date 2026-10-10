#!/usr/bin/env bash
# scripts/lint_check.sh
# Runs the full ZUNTENIUM linting suite: ruff + mypy + docstring check.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXIT_CODE=0

echo "═══════════════════════════════════════════"
echo " ZUNTENIUM — Lint & Type Check"
echo "═══════════════════════════════════════════"

# ── ruff ──────────────────────────────────────────────────────────────────
echo ""
echo "[1/3] ruff linter..."
if ruff check "${PROJECT_ROOT}/zuntenium/" "${PROJECT_ROOT}/tests/"; then
    echo "  ✓ ruff: no issues"
else
    echo "  ✗ ruff: issues found"
    EXIT_CODE=1
fi

# ── ruff format check ─────────────────────────────────────────────────────
echo ""
echo "[2/3] ruff format check..."
if ruff format --check "${PROJECT_ROOT}/zuntenium/" "${PROJECT_ROOT}/tests/"; then
    echo "  ✓ format: OK"
else
    echo "  ✗ format: run 'make format' to fix"
    EXIT_CODE=1
fi

# ── mypy ──────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] mypy type checker..."
if mypy "${PROJECT_ROOT}/zuntenium/" --ignore-missing-imports; then
    echo "  ✓ mypy: no type errors"
else
    echo "  ✗ mypy: type errors found"
    EXIT_CODE=1
fi

echo ""
echo "═══════════════════════════════════════════"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo " ✓ All checks passed."
else
    echo " ✗ One or more checks failed."
fi
echo "═══════════════════════════════════════════"

exit "${EXIT_CODE}"

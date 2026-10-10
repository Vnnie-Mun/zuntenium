.PHONY: install dev-install build-bindings test test-unit test-integration \
        benchmark lint format typecheck clean docs help

# ── Settings ────────────────────────────────────────────────────────────────
PYTHON      := python3
PIP         := pip
PYTEST      := pytest
RUFF        := ruff
MYPY        := mypy
CMAKE_BUILD := build/_cmake

# ── Help ────────────────────────────────────────────────────────────────────
help:
	@echo "ZUNTENIUM v0.1 — Makefile targets"
	@echo ""
	@echo "  install          Install package (production)"
	@echo "  dev-install      Install package (editable + dev deps)"
	@echo "  build-bindings   Compile C++ pybind11 extension"
	@echo "  test             Run full test suite with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  benchmark        Run benchmark suite"
	@echo "  lint             Run ruff linter"
	@echo "  format           Auto-format code with ruff"
	@echo "  typecheck        Run mypy type checker"
	@echo "  clean            Remove build artifacts"
	@echo "  docs             Build documentation"

# ── Install ─────────────────────────────────────────────────────────────────
install:
	$(PIP) install .

dev-install:
	$(PIP) install -e ".[dev]"
	pre-commit install

# ── C++ Bindings ─────────────────────────────────────────────────────────────
build-bindings:
	@bash scripts/build_bindings.sh

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ --cov=zuntenium --cov-report=term-missing --cov-report=xml -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

benchmark:
	$(PYTEST) tests/benchmarks/ --benchmark-only --benchmark-json=benchmark_results.json

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	$(RUFF) check zuntenium/ tests/
	@bash scripts/lint_check.sh

format:
	$(RUFF) format zuntenium/ tests/
	$(RUFF) check --fix zuntenium/ tests/

typecheck:
	$(MYPY) zuntenium/

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf $(CMAKE_BUILD)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.so" -delete

# ── Docs ─────────────────────────────────────────────────────────────────────
docs:
	@echo "Building docs..."
	cd docs && make html 2>/dev/null || echo "Sphinx not configured yet — see docs/architecture/"

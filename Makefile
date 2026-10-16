.PHONY: install dev-install build-bindings test test-unit test-integration \
        benchmark lint format typecheck clean docs help doctor

# ── Cross-platform note ──────────────────────────────────────────────────────
# This Makefile requires GNU make and bash — Linux and macOS only.
# On Windows, use scripts/build.py directly:
#   py scripts/build.py <command>
# All targets below delegate to scripts/build.py for portability (U2, U3, U7).

# Prefer the Python running inside an active venv; fall back to python3 / python.
PYTHON := $(or $(shell command -v python3 2>/dev/null), python)

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ZUNTENIUM QuFlow — Make targets (Linux / macOS)"
	@echo "On Windows: py scripts/build.py <command>"
	@echo ""
	@echo "  install          Install package (production)"
	@echo "  dev-install      Install package (editable + dev deps)"
	@echo "  build-bindings   Compile C++ pybind11 extension (.so / .pyd / .dylib)"
	@echo "  test             Run full test suite with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint             Run ruff linter"
	@echo "  format           Auto-format code with ruff"
	@echo "  typecheck        Run mypy type checker"
	@echo "  clean            Remove build artefacts"
	@echo "  doctor           Print platform diagnostics"
	@echo "  docs             Build documentation"
	@echo ""

# ── Install ──────────────────────────────────────────────────────────────────
install:
	$(PYTHON) scripts/build.py install

dev-install:
	$(PYTHON) scripts/build.py dev-install

# ── C++ Bindings ─────────────────────────────────────────────────────────────
# Delegates to scripts/build.py which handles .so (Linux), .dylib (macOS),
# and .pyd (Windows) — fixing CL-17. build_bindings.sh kept for reference.
build-bindings:
	$(PYTHON) scripts/build.py bindings

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) scripts/build.py test

test-unit:
	$(PYTHON) scripts/build.py test-unit

test-integration:
	$(PYTHON) scripts/build.py test-integration

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	$(PYTHON) scripts/build.py lint

format:
	$(PYTHON) scripts/build.py format

typecheck:
	$(PYTHON) scripts/build.py typecheck

# ── Clean ─────────────────────────────────────────────────────────────────────
# Uses scripts/build.py clean (pure Python, works on all OS).
clean:
	$(PYTHON) scripts/build.py clean

# ── Doctor ────────────────────────────────────────────────────────────────────
doctor:
	$(PYTHON) scripts/build.py doctor

# ── Docs ──────────────────────────────────────────────────────────────────────
docs:
	@echo "Building docs..."
	cd docs && make html 2>/dev/null || echo "Sphinx not configured yet — see docs/architecture/"

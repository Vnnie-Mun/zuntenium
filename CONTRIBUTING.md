# CONTRIBUTING to ZUNTENIUM

Thank you for contributing! Please read this guide before submitting a PR.

## Development Setup

```bash
git clone https://github.com/zuntenium/zuntenium.git
cd zuntenium
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

We enforce:
- **ruff** for linting (PEP 8, import sorting)
- **black** for formatting (line length 99)
- **mypy** for type annotations (strict mode for core modules)

```bash
make lint       # run ruff + mypy
make format     # run black + ruff --fix
```

## Writing Tests

All new features **must** include tests:

- **Unit tests** in `tests/unit/` — pure logic, no I/O
- **Integration tests** in `tests/integration/` — backend + circuit execution
- **Benchmarks** in `tests/benchmarks/` — performance regression guards

```bash
make test              # full suite
make test-unit         # unit only
pytest tests/ -v -k "test_bell"  # single test
```

## Adding a New Gate

1. Add the gate class to the appropriate file in `zuntenium/quantum_core/gates/`
2. Export it from `zuntenium/quantum_core/gates/__init__.py`
3. Add a gate method to `QuantumCircuit` in `zuntenium/circuit/circuit.py`
4. Add to `_QASM2_GATE_MAP` in `qasm_serializer.py` and `qasm_parser.py`
5. Add unit tests to `tests/unit/test_gates.py`
6. Add `@pytest.mark.parametrize` entry in `TestGateUnitarity`

## Adding a Transpiler Pass

1. Create `zuntenium/transpiler/passes/<category>/<pass_name>.py`
2. Inherit from `TranspilerPass` and implement `run(ir, **kwargs) -> QuantumIR`
3. Set `name`, `description`, and optionally `requires`
4. Add to `_default_passes()` in `pipeline.py` at the appropriate level
5. Write unit tests in `tests/unit/test_transpiler.py`

## Pull Request Checklist

- [ ] Tests pass: `make test`
- [ ] Linting passes: `make lint`
- [ ] Type annotations added for all public functions
- [ ] Docstrings updated
- [ ] CHANGELOG entry added under `[Unreleased]`
- [ ] No breaking changes to stable public API (without version bump plan)

## Commit Message Format

```
<type>(<scope>): <short summary>

Types: feat | fix | refactor | test | docs | perf | ci | chore
Scope: gates | circuit | transpiler | backend | hybrid | utils | api | bindings

Example:
feat(transpiler): add commutation-aware gate cancellation pass
```

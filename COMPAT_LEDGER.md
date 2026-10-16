# COMPAT_LEDGER.md — Zuntenium QuFlow Compatibility Defect Ledger

> Last updated: 2026-10-16 | Evidence baseline: audit of `main` HEAD  
> Format: `ID · Cell · Symptom · Root cause (U-ref) · Severity · Status`

---

## How to Use This File

- **ID**: Stable reference (CL-xx) — never reuse a closed ID.
- **Cell**: OS / Python / Arch / Install method — or "All" if universal.
- **Severity**: HIGH = blocks use on that cell; MEDIUM = degrades or misleads; LOW = cosmetic or edge case.
- **Status**: OPEN | FIXED `commit` | DOCUMENTED (no code fix; behaviour documented instead) | WONTFIX `reason`

---

## Legend

| Status | Meaning |
|--------|---------|
| ✅ FIXED | Code change landed; commit hash given |
| 📄 DOCUMENTED | Behaviour is a platform limitation; docs updated with workaround |
| 🔲 OPEN | Work not yet started |
| 🔶 IN PROGRESS | Being worked on |

---

## Active Items

| ID | Cell | Symptom | U-ref | Severity | Status |
|----|------|---------|-------|----------|--------|
| CL-15 | All | `pickle` used for model serialization in `ai_engine/` — unsafe across Python versions and platforms on any network/storage boundary | U13 | MEDIUM | 🔲 OPEN — Phase 7 |
| CL-16 | Windows | `Makefile` uses `rm -rf`, `find -exec`, `bash`, `python3` — all POSIX-only; `make` not available by default on Windows | U2/U7 | MEDIUM | 🔲 OPEN — Phase 4 (superceded by `scripts/build.py`) |
| CL-17 | Windows | `build_bindings.sh` only searches for `*.so`; Windows extensions are `*.pyd` (`_zuntenium_core.cpXX-win_amd64.pyd`) | U5 | MEDIUM | 🔲 OPEN — Phase 3 (superceded by `scripts/build.py`) |
| CL-20 | All | `requirements.txt` version pins drift from `pyproject.toml` (e.g. `fastapi>=0.100` vs `>=0.110`) — risk of wrong version resolving | U4 | LOW | 🔲 OPEN — Phase 9 |

---

## Closed Items

| ID | Cell | Symptom | U-ref | Severity | Status |
|----|------|---------|-------|----------|--------|
| CL-01 | Windows | CI never ran on `windows-latest`; no Windows test evidence | U1 | HIGH | ✅ FIXED `50653a0` |
| CL-02 | All shells | README/Makefile/CI use bare `pip`; resolves wrong interpreter on multi-Python systems | U2/U3 | HIGH | ✅ FIXED `50653a0` |
| CL-03 | All shells | README uses bare `pytest`; fails on systems where only `python3 -m pytest` works | U2/U3 | MEDIUM | ✅ FIXED `50653a0` |
| CL-04 | All shells | README uses bare `python debug_all.py`; `python` not available on many Linux/macOS | U3 | MEDIUM | ✅ FIXED `50653a0` |
| CL-05 | Same host | Grafana (`localhost:3000`) and Next.js UI (`localhost:3000`) documented at same port — impossible to run together | U10 | HIGH | ✅ FIXED `50653a0` — Grafana moved to `:3050` |
| CL-06 | All | `docker compose up -d` assumes Compose v2 plugin; standalone `docker-compose` behaves differently | U10 | MEDIUM | ✅ FIXED `50653a0` — v2 requirement documented |
| CL-07 | Windows native | RQ workers require `os.fork()`; not available natively on Windows | U9 | HIGH | 📄 DOCUMENTED `50653a0` — "use Docker Desktop or WSL2" |
| CL-08 | Windows native | Redis server has no official native Windows build since 2016 | U9 | MEDIUM | 📄 DOCUMENTED `50653a0` — "use Docker Desktop or WSL2" |
| CL-09 | macOS | `pip install "zuntenium[full]"` includes GPU extra; env marker gates it correctly but was undocumented | U6 | LOW | ✅ FIXED `50653a0` — README clarifies `[all]` vs `[full]` |
| CL-10 | All (Windows risk) | `open()` without `encoding=` in 7 locations — reads/writes at platform default encoding, not UTF-8 | U8 | MEDIUM | ✅ FIXED `50653a0` |
| CL-11 | All (Windows risk) | `read_text()`/`write_text()` without `encoding=` in 11 locations | U8 | MEDIUM | ✅ FIXED `50653a0` |
| CL-12 | Windows | `_CRED_FILE.chmod(0o600)` is a no-op on Windows; no explicit platform guard | U7/U14 | LOW | ✅ FIXED `50653a0` — wrapped in `try/except OSError` |
| CL-13 | Cross-partition | `pdf.rename(out_path)` in `latex_drawer.py` fails when temp dir and output are on different partitions/drives | U7 | LOW | ✅ FIXED `50653a0` — replaced with `shutil.copy2()` |
| CL-14 | Windows legacy console | Unicode `✓` `✗` `═` in CLI output raises `UnicodeEncodeError` on CP850/CP1252 consoles | U8 | MEDIUM | ✅ FIXED `50653a0` — ASCII fallback via `sys.stdout.encoding` |
| CL-18 | All | "549/549 passing" badge claimed no platform context — meaningless as evidence | U1/U15 | LOW | ✅ FIXED `50653a0` — badge now reads "(Ubuntu + macOS)" |
| CL-19 | All | No `engines` field in `zuntenium-ui/package.json`; Node version requirement invisible | U11 | MEDIUM | ✅ FIXED `50653a0` — `engines: {node: ">=18.17.0"}` |
| CL-21 | All | `docker-compose.prod.yml` had deprecated `version: "3.9"` key — ignored by Compose v2, confuses v1 | U10 | LOW | ✅ FIXED `50653a0` — key removed |

---

## Adding New Items

```
| CL-xx | <OS / Python / Arch> | <symptom> | U-ref | HIGH/MEDIUM/LOW | 🔲 OPEN |
```

Use the next available ID. Never recycle IDs. Link to the CI run or transcript that demonstrates the failure.

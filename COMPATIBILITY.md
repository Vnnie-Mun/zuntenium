# ZUNTENIUM QuFlow — COMPATIBILITY.md
# Phase 10 / U1: Final compatibility certification record.
# Evidence expires after 90 days. Stale cells are marked UNTESTED.
# Source: SUPPORT_MATRIX.md (tiered matrix), COMPAT_LEDGER.md (issue tracking),
#         HUMAN_SIGNOFF_LIST.md (manual checks).
# Generated-equivalent: CI archives `zqc doctor --json` output per run.
# Last reviewed: 2026-10-16

# Table of Contents
# 1. Certified Cells (Tier 1) — fully automated CI per commit
# 2. Supported Cells (Tier 2) — tested on a schedule
# 3. Experimental Cells (Tier 3) — best-effort / community
# 4. Unsupported Combinations (with reasons)
# 5. Port Assignment Table
# 6. Cloud Stack OS Support Statement
# 7. Open Ledger Items
# 8. 12-Month Maintenance Calendar
# 9. Evidence Archive Reference

---

## 1. Certified Cells (Tier 1)
> CI runs on every commit to `main` and every pull request.
> Evidence: `.github/workflows/ci.yml` matrix. All cells must pass before merge.

| OS            | Version   | Arch  | Python | Install method | Shell      | Status     | Last pass      |
|---------------|-----------|-------|--------|----------------|------------|------------|----------------|
| Ubuntu        | 24.04 LTS | x86_64| 3.10   | pip (venv)     | bash       | CERTIFIED  | CI — automated |
| Ubuntu        | 24.04 LTS | x86_64| 3.11   | pip (venv)     | bash       | CERTIFIED  | CI — automated |
| Ubuntu        | 24.04 LTS | x86_64| 3.12   | pip (venv)     | bash       | CERTIFIED  | CI — automated |
| macOS         | 14 (ARM)  | arm64 | 3.10   | pip (venv)     | zsh        | CERTIFIED  | CI — automated |
| macOS         | 14 (ARM)  | arm64 | 3.11   | pip (venv)     | zsh        | CERTIFIED  | CI — automated |
| macOS         | 14 (ARM)  | arm64 | 3.12   | pip (venv)     | zsh        | CERTIFIED  | CI — automated |
| Windows       | Server 22 | x64   | 3.11   | pip (venv)     | PowerShell | CERTIFIED  | CI — automated |
| Windows       | Server 22 | x64   | 3.12   | pip (venv)     | PowerShell | CERTIFIED  | CI — automated |

> [!NOTE]
> "Certified" means the full CI suite (lint, mypy, pytest) passes in an isolated venv
> starting from stock Python on that runner image. No manual steps.

---

## 2. Supported Cells (Tier 2)
> Tested on a schedule (weekly CI and/or quarterly manual run). May lag behind Tier 1.

| OS          | Version | Arch   | Python | Install method | Status         | Target frequency |
|-------------|---------|--------|--------|----------------|----------------|-----------------|
| Ubuntu      | 22.04   | x86_64 | 3.11   | pip            | SUPPORTED      | Weekly CI       |
| Fedora      | 40      | x86_64 | 3.12   | pip            | [UNVERIFIED]   | Quarterly       |
| Debian      | 12      | x86_64 | 3.11   | pip            | [UNVERIFIED]   | Quarterly       |
| Alpine      | 3.20    | x86_64 | 3.12   | pip (musl)     | [UNVERIFIED]   | Quarterly       |
| macOS       | 13 (x64)| x86_64 | 3.11   | pip            | [UNVERIFIED]   | Quarterly       |
| Windows     | 11      | x64    | 3.11   | pip + py launch| [UNVERIFIED]   | Quarterly       |
| WSL2        | Ubuntu  | x86_64 | 3.12   | pip            | [UNVERIFIED]   | Quarterly       |

> [!IMPORTANT]
> `[UNVERIFIED]` cells have been designed for but not yet tested on that exact runner.
> They will be upgraded to SUPPORTED once a passing run is recorded in CI with evidence.

---

## 3. Experimental Cells (Tier 3)
> Best-effort; use at own risk. No CI currently runs on these.

| OS            | Version | Arch   | Notes                                           |
|---------------|---------|--------|-------------------------------------------------|
| Arch Linux    | rolling | x86_64 | No scheduled testing; community reports welcome |
| Windows ARM64 | 11      | arm64  | No ARM64 Windows runner available yet [UNVERIFIED] |
| macOS arm64   | 15 (RC) | arm64  | Testing added when GA; current: 14 only         |
| Linux aarch64 | Ubuntu  | arm64  | Docker multi-arch images built; host install untested |

---

## 4. Unsupported Combinations (with reasons)
| Combination                      | Reason                                                                                       |
|----------------------------------|----------------------------------------------------------------------------------------------|
| Python < 3.10                    | `match` statement, `tomllib`, `pathlib` features used; end-of-life upstream                  |
| Python 3.13 free-threaded (3.13t)| scikit-learn and torch wheels not yet published for `cp313t`; no CI evidence                 |
| macOS native CUDA                | CUDA not supported on macOS by NVIDIA. GPU extra installs but silently uses CPU fallback.    |
| Redis native Windows             | Redis has no official Windows build; use Docker or WSL2 (see Cloud Stack statement §6)       |
| RQ workers native Windows        | RQ uses `os.fork()`; Windows uses spawn; native Windows workers not supported (use Docker)   |
| 32-bit Python (x86)              | NumPy drops 32-bit Windows; native extension requires LP64 ABI                               |
| Python free-threaded + sklearn   | sklearn not wheel-published for free-threaded; see CL-20 in COMPAT_LEDGER.md               |
| Podman (rootless)                | Tested as best-effort; multi-arch emulation under rootless Podman unverified                 |

---

## 5. Port Assignment Table
> All host-side ports are configurable via `.env` file (see `.env.example`).
> Grafana deliberately uses 3050 (not 3000) to avoid conflict with the Next.js UI.

| Service         | Default host port | Container port | Config variable    | Notes                        |
|-----------------|:-----------------:|:--------------:|--------------------|-----------------------------|
| Caddy (HTTP)    | 80                | 80             | —                  | Reverse proxy                |
| Caddy (HTTPS)   | 443               | 443            | —                  | TLS termination              |
| Next.js UI      | **3000**          | 3000           | `UI_PORT`          | QCanvas frontend             |
| Collab WS       | 3001              | 3001           | —                  | Real-time collaboration      |
| FastAPI gateway | 8000              | 8000           | `API_PORT`         | REST + WebSocket             |
| PostgreSQL      | 5432              | 5432           | `POSTGRES_PORT`    | Persistent state             |
| Redis           | 6379              | 6379           | `REDIS_PORT`       | Job queue + cache            |
| Prometheus      | 9090              | 9090           | `PROMETHEUS_PORT`  | Metrics scraper              |
| Grafana         | **3050** ← NOT 3000 | 3000        | `GRAFANA_PORT`     | Monitoring dashboard         |
| RQ Dashboard    | 9181              | 9181           | `RQ_DASHBOARD_PORT`| Worker queue monitor         |

---

## 6. Cloud Stack OS Support Statement
> Applies to: FastAPI, RQ workers, Redis, PostgreSQL, Prometheus, Grafana.
> Verified source dates should be updated quarterly; entries marked [UNVERIFIED] have not been
> checked against current official documentation since this file was created (2026-10-16).

| Component    | Native Linux | Docker/WSL2 (Windows) | Native macOS        | Native Windows     |
|--------------|:------------:|:---------------------:|:--------------------:|:------------------:|
| FastAPI      | ✅ Certified | ✅ Certified (Docker) | ✅ Certified         | ⚠️ Supported†      |
| RQ workers   | ✅ Certified | ✅ Docker only         | ⚠️ Experimental‡    | ❌ Unsupported     |
| Redis        | ✅ Certified | ✅ Docker only         | ✅ Homebrew/Docker   | ❌ No official build |
| PostgreSQL   | ✅ Certified | ✅ Docker              | ✅ Homebrew/Docker   | ✅ Official installer [UNVERIFIED] |
| Prometheus   | ✅ Certified | ✅ Docker              | ✅ Docker            | ✅ Docker          |
| Grafana      | ✅ Certified | ✅ Docker              | ✅ Docker            | ✅ Docker          |

† FastAPI on native Windows: tested in CI (GitHub Actions `windows-latest`).
  No `uvicorn --workers` multiprocessing on Windows without spawn guard.
‡ RQ on macOS: `multiprocessing` uses `spawn` by default on Python ≥ 3.12;
  RQ's `Worker` class uses `os.fork()` internally; behaviour on macOS is
  **best-effort** and containers (Docker Desktop) are the recommended path.
  [UNVERIFIED — check RQ release notes for spawn support status]

> [!WARNING]
> Do not deploy RQ workers natively on Windows or rely on fork-based concurrency
> outside Linux. Use the provided Compose stack (`docker-compose.prod.yml`) instead.

---

## 7. Open Ledger Items
> Full details: [COMPAT_LEDGER.md](COMPAT_LEDGER.md)

| ID    | Severity | Summary                                                  | U-ref | Status  |
|-------|----------|----------------------------------------------------------|-------|---------|
| CL-15 | Medium   | pickle in ai_engine — boundary guards added (Phase 7)   | U13   | Mitigated — local disk only, noqa + path validation added |
| CL-20 | Low      | Version pin drift requirements.txt vs pyproject.toml    | U4    | Open — Phase 9 |
| CL-22 | Low      | Missing type annotations in ai_optimizer_part2.py       | U3    | Open — Phase 9 |

---

## 8. 12-Month Maintenance Calendar
> Scheduled activities to keep the matrix current and evidence fresh.

| Frequency   | Activity                                                                                | Owner       |
|-------------|-----------------------------------------------------------------------------------------|-------------|
| Per commit   | CI: full pytest on all Tier-1 cells (Linux, macOS, Windows × Python 3.10–3.12)        | Automated   |
| Weekly       | Scheduled CI: Tier-2 cells (Ubuntu 22.04, pre-release Python)                          | Automated   |
| Weekly       | UI CI: npm ci + build + dev server smoke test on Linux/macOS/Windows (Node 18 and 20)  | Automated   |
| Monthly      | Review HUMAN_SIGNOFF_LIST.md for incomplete items; nudge responsible parties            | Maintainer  |
| Quarterly    | EOL calendar review: check Python, Node, Ubuntu, Fedora, Debian EOL dates vs matrix    | Maintainer  |
| Quarterly    | Evidence expiry sweep: mark UNTESTED cells where last pass >90 days old                | Maintainer  |
| Quarterly    | Tier-2 manual run on platforms not covered by automated CI (Fedora, Alpine, macOS Intel)| Maintainer |
| Quarterly    | Dependency audit: `pip index versions zuntenium`; verify wheel availability per cell   | Maintainer  |
| Bi-annual    | Channel parity review: install same version from each active channel; compare smoke output | Maintainer |
| Annual       | Full matrix review: add/remove cells based on upstream EOL and observed user demand     | Maintainer  |
| Annual       | Security review of install and credential paths (see HUMAN_SIGNOFF_LIST.md)            | Human only  |
| On each release | Run release gate checklist (RELEASE_CHECKLIST.md) before publishing                 | Maintainer  |

---

## 9. Evidence Archive Reference
> CI run artifacts are stored as GitHub Actions workflow artifacts for 90 days.
> Each `zqc doctor --json` output (attached to every CI run) provides:
> - Python version, architecture, free-threaded flag
> - OS, libc, shell, locale, stdout encoding
> - Writable directory status, PATH gap detection
> - Docker + Compose version, Node + npm version
> - Core dependency versions

To reproduce any cell locally:
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install zuntenium
python -m zuntenium doctor --json
```

```powershell
# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install zuntenium
python -m zuntenium doctor --json
```

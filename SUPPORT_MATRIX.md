# SUPPORT_MATRIX.md — Zuntenium QuFlow Support Matrix

> Version: 0.1-baseline | Date: 2026-10-16 | Evidence: CI + manual audit  
> **Tiers**: Certified = passes in CI every push · Supported = tested weekly · Experimental = best-effort · Unsupported = stated + documented alternative

---

## Python SDK (`pip install zuntenium`)

| OS | Version | Arch | Python | Install method | Tier | Evidence | Notes |
|----|---------|------|--------|---------------|------|---------|-------|
| Ubuntu | 22.04 LTS | x86_64 | 3.10, 3.11, 3.12 | pip | **Certified** | CI: `ubuntu-latest` every push | |
| Ubuntu | 24.04 LTS | x86_64 | 3.12 | pip | **Supported** | [UNVERIFIED — runner not yet added] | Needs scheduled job |
| macOS | 14 (Sonoma) | arm64 (Apple Silicon) | 3.11, 3.12 | pip | **Certified** | CI: `macos-latest` every push | |
| macOS | 13 (Ventura) | x86_64 (Intel) | 3.11, 3.12 | pip | **Supported** | [UNVERIFIED] | Needs scheduled job |
| Windows | 11 | x86_64 | 3.11, 3.12 | pip | **Certified** | CI: `windows-latest` added 50653a0 | ⚠️ First run evidence pending |
| Windows | 10 | x86_64 | 3.11, 3.12 | pip | **Supported** | [UNVERIFIED] | |
| Debian | 12 (Bookworm) | x86_64 | 3.11 | pip | **Supported** | [UNVERIFIED] | Container-based test needed |
| Ubuntu | 22.04 LTS | aarch64 | 3.11, 3.12 | pip | **Experimental** | [UNVERIFIED] | ARM64 runner needed |
| Alpine | 3.19 | x86_64 | 3.12 | pip | **Experimental** | [UNVERIFIED] | musl libc; `psycopg2-binary` won't work — use `psycopg2` |
| Windows | 11 | ARM64 | 3.12 | pip | **Experimental** | [UNVERIFIED] | Very low user count |
| WSL2 (Ubuntu 22.04) | — | x86_64 | 3.11, 3.12 | pip | **Supported** | [UNVERIFIED] | Linux behaviour inside Windows |

### Python Version Support

| Python | Status | EOL | Notes |
|--------|--------|-----|-------|
| 3.10 | Supported | 2026-10 [UNVERIFIED — check python.org] | Dropping after EOL |
| 3.11 | Certified | 2027-10 [UNVERIFIED] | Primary |
| 3.12 | Certified | 2028-10 [UNVERIFIED] | Primary |
| 3.13 | Experimental | 2029-10 [UNVERIFIED] | Free-threaded build not yet tested |
| ≤3.9 | **Unsupported** | Past EOL | |

---

## Cloud Stack (Docker Compose)

> Requires: Docker 24+ with **Compose v2 plugin** (`docker compose`, not `docker-compose`)

| Component | Linux (native) | macOS (Docker Desktop) | Windows (Docker Desktop + WSL2) | Notes |
|-----------|---------------|----------------------|-------------------------------|-------|
| FastAPI API gateway | **Certified** | **Supported** | **Supported** | Containerised |
| RQ Workers | **Certified** | **Supported** (Docker) | **Supported** (Docker/WSL2 only) | **Unsupported natively** — `fork()` required |
| Redis 7 | **Certified** | **Supported** (Docker) | **Supported** (Docker/WSL2 only) | No official native Windows build |
| PostgreSQL 15 | **Certified** | **Supported** (Docker) | **Supported** (Docker) | |
| Prometheus | **Certified** | **Supported** | **Supported** | Monitoring profile |
| Grafana | **Certified** | **Supported** | **Supported** | Monitoring profile; port **3050** (not 3000) |
| Full stack (Compose) | **Certified** | **Supported** | **Supported** | Compose v2 required |

### Port Table (all configurable via `.env`)

| Service | Default Host Port | Container Port | Profile |
|---------|------------------|---------------|---------|
| Caddy (HTTP) | 80 | 80 | default |
| Caddy (HTTPS) | 443 | 443 | default |
| Next.js UI | 3000 | 3000 | default |
| Collab WS | 3001 | 3001 | default |
| FastAPI | 8000 | 8000 | default |
| PostgreSQL | 5432 | 5432 | default |
| Redis | 6379 | 6379 | default |
| Prometheus | 9090 | 9090 | monitoring |
| Grafana | **3050** | 3000 | monitoring |
| RQ Dashboard | 9181 | 9181 | monitoring |

---

## Next.js UI (`zuntenium-ui`)

| OS | Node version | Tier | Notes |
|----|-------------|------|-------|
| Linux (Ubuntu 22.04) | 18.x, 20.x, 22.x | **Certified** | [UNVERIFIED — UI CI not yet added] |
| macOS 14 Apple Silicon | 20.x | **Supported** | [UNVERIFIED] |
| Windows 11 | 20.x | **Supported** | [UNVERIFIED] |

> **Minimum Node**: 18.17.0 (Next.js 15 requirement — [UNVERIFIED, check nextjs.org/docs])  
> **Minimum npm**: 9.0.0

---

## GPU Acceleration

| Platform | CUDA | ROCm | Apple Metal | Status |
|----------|------|------|-------------|--------|
| Linux x86_64 + CUDA 12 | ✅ via `[gpu]` extra | ❌ [UNVERIFIED] | N/A | **Experimental** — needs GPU runner |
| Windows x64 + CUDA 12 | [UNVERIFIED] | N/A | N/A | **Unsupported** — no env marker yet |
| macOS (any) | N/A | N/A | ❌ Not implemented | **Unsupported** — CPU fallback active |
| Linux aarch64 | [UNVERIFIED] | N/A | N/A | **Unsupported** |

---

## Explicit Unsupported Combinations

| What | Reason | Alternative |
|------|--------|------------|
| GPU on macOS | CUDA not available; Apple Metal not implemented | CPU fallback (automatic) |
| Native RQ workers on Windows | `os.fork()` unavailable | Docker Desktop or WSL2 |
| Native Redis server on Windows | No official Windows build | Docker Desktop or WSL2 |
| Python ≤3.9 | EOL or approaching; f-string and typing syntax used | Upgrade to 3.11+ |
| Free-threaded Python 3.13 | Not tested | Experimental; may work |
| Alpine musl with `psycopg2-binary` | Binary wheel not built for musl | Use `psycopg2` source package |
| 32-bit architectures | Not targeted | N/A |

---

## End-of-Life Calendar

> All dates marked [UNVERIFIED] must be confirmed against official lifecycle pages before each release.

| Software | Version | EOL Date | Source |
|----------|---------|----------|--------|
| Python | 3.10 | Oct 2026 | [UNVERIFIED — python.org/downloads] |
| Python | 3.11 | Oct 2027 | [UNVERIFIED] |
| Python | 3.12 | Oct 2028 | [UNVERIFIED] |
| Ubuntu | 22.04 LTS | Apr 2027 (standard) | [UNVERIFIED — ubuntu.com/about/release-cycle] |
| Ubuntu | 24.04 LTS | Apr 2029 (standard) | [UNVERIFIED] |
| Windows | 10 | Oct 2025 | [UNVERIFIED — microsoft.com/lifecycle] |
| Windows | 11 | Oct 2026 | [UNVERIFIED] |
| Node.js | 18.x LTS | Apr 2025 | [UNVERIFIED — nodejs.org/en/about/previous-releases] |
| Node.js | 20.x LTS | Apr 2026 | [UNVERIFIED] |
| Node.js | 22.x LTS | Apr 2027 | [UNVERIFIED] |

---

## Evidence Archive

Each CI run that constitutes evidence for a Certified tier cell must be linked here. Stale evidence (older than 90 days with no newer run) is marked UNTESTED.

| Cell | Evidence link | Date | Status |
|------|--------------|------|--------|
| Ubuntu 22.04 / Python 3.10-3.12 | GitHub Actions — `ci.yml` | 2026-10-16 | ✅ Certified |
| macOS 14 arm64 / Python 3.11-3.12 | GitHub Actions — `ci.yml` | 2026-10-16 | ✅ Certified |
| Windows 11 x64 / Python 3.11-3.12 | GitHub Actions — `ci.yml` | 2026-10-16 | ⚠️ First run pending 50653a0 |
| All other cells | — | — | [UNVERIFIED] |

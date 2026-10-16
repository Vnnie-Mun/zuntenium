# HUMAN_SIGNOFF_LIST.md — Zuntenium QuFlow Release Sign-off

> Tasks that **only a human can complete**. Do not mark any item here as done automatically.  
> Each item must be completed by a named person with a date before a release is certified.

---

## Code Signing & Notarization (U12)

| # | Item | Owner | Status | Date |
|---|------|-------|--------|------|
| S1 | Obtain Apple Developer Program membership and signing certificate | — | ⬜ TODO | — |
| S2 | Notarize any macOS downloadable `.pkg` or `.dmg` installer using `notarytool` | — | ⬜ TODO | — |
| S3 | Obtain Microsoft Authenticode code-signing certificate for Windows installers | — | ⬜ TODO | — |
| S4 | Sign any Windows `.exe` / `.msi` installer | — | ⬜ TODO | — |
| S5 | Store certificates in a hardware token or cloud secrets manager (never in plaintext) | — | ⬜ TODO | — |

## PyPI and Distribution Accounts (U12)

| # | Item | Owner | Status | Date |
|---|------|-------|--------|------|
| D1 | Configure PyPI Trusted Publishing (OIDC via GitHub Actions) for `zuntenium` — verify current mechanism at packaging.python.org | — | ⬜ TODO | — |
| D2 | Set up conda-forge feedstock — evaluate demand first (ADR required) | — | ⬜ TODO | — |
| D3 | Create Docker Hub / GHCR repository with access control | — | ⬜ TODO | — |
| D4 | Register Homebrew tap or evaluate submitting to homebrew-core | — | ⬜ TODO | — |
| D5 | Set up winget/Scoop/Chocolatey packages — ADR required to justify maintenance cost | — | ⬜ TODO | — |

## Security Review (U14)

| # | Item | Reviewer | Status | Date |
|---|------|---------|--------|------|
| SR1 | Security review of credential storage in `~/.zuntenium/credentials.json`; confirm headless fallback is safe for production servers | — | ⬜ TODO | — |
| SR2 | Review `_CRED_FILE.chmod(0o600)` fallback on Windows — confirm OS user-profile scoping is adequate | — | ⬜ TODO | — |
| SR3 | Confirm no secrets are logged or printed in any log level | — | ⬜ TODO | — |
| SR4 | Review Docker Compose production stack for least-privilege defaults | — | ⬜ TODO | — |

## Legal Review (U12)

| # | Item | Reviewer | Status | Date |
|---|------|---------|--------|------|
| L1 | Legal review of any bundled third-party libraries (license compatibility) | — | ⬜ TODO | — |
| L2 | Legal review of redistributing GPU runtime components (CUDA, cuDNN) — verify NVIDIA EULA | — | ⬜ TODO | — |
| L3 | Export control review if the package is to be published internationally | — | ⬜ TODO | — |

## MANUAL Hardware Tests (U6, U13)

| # | Item | Tester | Tool version | Date |
|---|------|--------|-------------|------|
| H1 | GPU test on Linux + NVIDIA GPU: `pip install "zuntenium[gpu]"` → smoke circuit | — | — | — |
| H2 | GPU test on Windows + NVIDIA GPU (CUDA 12): install + smoke circuit | — | — | — |
| H3 | Apple Silicon MPS test (if/when implemented): `[full]` install → smoke circuit | — | — | — |
| H4 | Full cloud stack on Docker Desktop for macOS (Apple Silicon) — bring-up from scratch | — | — | — |
| H5 | Full cloud stack on Docker Desktop for Windows + WSL2 — bring-up from scratch | — | — | — |

## CI Runners to Provision (U1)

| # | Runner | Purpose | Provisioned by | Status |
|---|--------|---------|---------------|--------|
| R1 | ARM64 Linux (GitHub larger runner or self-hosted) | Supported tier for aarch64 | — | ⬜ TODO |
| R2 | macOS 13 x86_64 (Intel) — scheduled job | Supported tier for Intel Mac | — | ⬜ TODO |
| R3 | GPU runner (Linux + NVIDIA) | Experimental GPU tier | — | ⬜ TODO |

## Maintenance Plan (Post-release, U1)

| Cadence | Task |
|---------|------|
| Weekly | Scheduled CI run against all Supported tier cells |
| Monthly | Review GitHub Actions runner version updates |
| Quarterly | Review OS and Python EOL calendar; retire support for EOL versions |
| Annually | Full channel review: evaluate demand vs maintenance cost for each distro channel |
| Per OS release | Test against new macOS, Windows, Ubuntu LTS candidates in the week of RC |
| Per Python RC | Test against Python release candidates 4 weeks before GA |
| Per Django major release | Update pyproject.toml constraint bounds if applicable |

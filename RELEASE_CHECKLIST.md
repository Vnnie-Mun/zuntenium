# ZUNTENIUM QuFlow — Release Gate Checklist
# Phase 10 / U12: Every item must be PASSED before tagging a release.
# Human-owned items (certificates, accounts, legal) are in HUMAN_SIGNOFF_LIST.md.
# DO NOT mark any human-owned items here.
#
# How to use:
#   1. Copy this file to a new GitHub Issue titled "Release vX.Y.Z gate"
#   2. Work through items in order; check each box only when verified, not assumed
#   3. Items marked [CI] are verified automatically by .github/workflows/release.yml (dry-run=true)
#   4. Items marked [HUMAN] require a person to verify — do not auto-check
#   5. Block publication until every box is checked; escalate blockers

---

## Pre-release Checklist

### Version and Changelog
- [ ] `pyproject.toml` → `project.version` matches the intended tag (`vX.Y.Z`)
- [ ] `CHANGELOG.md` (or equivalent) entry written for this version
- [ ] All commits since last release reviewed for breaking changes
- [ ] Breaking changes (if any) approved by maintainer and shimmed or documented

### Code Quality [CI]
- [ ] All Tier-1 CI cells pass: ubuntu-latest, macos-latest, windows-latest × Python 3.10–3.12
- [ ] Ruff lint: zero errors on all changed files
- [ ] mypy: zero new errors introduced versus last release baseline
- [ ] No new unexplained entries in COMPAT_LEDGER.md

### Tests [CI]
- [ ] Full pytest suite passes on all Tier-1 cells (exit code 0, zero failures)
- [ ] UI CI passes: `npm ci`, `npm run build`, dev server smoke test on Linux/macOS/Windows
- [ ] Release workflow dry-run passes (trigger `.github/workflows/release.yml` with `dry_run=true`)

### Build Artifacts [CI]
- [ ] `python -m build` produces both sdist (`.tar.gz`) and wheel (`.whl`) without warnings
- [ ] Wheel filename platform tag matches intended targets (check with `python -m wheel tags dist/*.whl`)
- [ ] `python -m zuntenium version` returns correct version from installed wheel
- [ ] `python -m zuntenium doctor --json` runs without errors on all Tier-1 cells

### Security and Provenance [CI + HUMAN]
- [ ] [CI] SHA256SUMS.txt generated and all hashes non-zero
- [ ] [CI] CycloneDX SBOM generated (zuntenium-sbom.json) and non-empty
- [ ] [HUMAN] SBOM reviewed: no unexpected new dependencies
- [ ] [HUMAN] Security review of install / credential paths completed (see HUMAN_SIGNOFF_LIST.md)
- [ ] No secret, token, or credential committed in this release's diff

### Serialization Safety [CI]
- [ ] Confirm no `pickle.load` from network-received data (grep evidence: `rg 'pickle.load' zuntenium/`)
- [ ] All cross-machine serialization uses JSON or `.pt` (PyTorch safe format)

### Documentation
- [ ] README.md install commands verified against the wheel being released
- [ ] COMPATIBILITY.md evidence table updated with latest CI run dates
- [ ] Port table in COMPATIBILITY.md matches `docker-compose.prod.yml` and `.env.example`
- [ ] SUPPORT_MATRIX.md updated if any cells changed tier since last release

### Docker and Containers [CI + HUMAN]
- [ ] [CI] Multi-arch Docker image (linux/amd64 + linux/arm64) builds without error
- [ ] [CI] `docker buildx imagetools inspect` shows both architectures in the manifest
- [ ] [HUMAN] Production Compose stack bring-up tested on Docker Desktop for macOS and/or Windows (see HUMAN_SIGNOFF_LIST.md)

### Channel Parity [CI partial + HUMAN]
- [ ] [CI] PyPI dry-run: `twine check dist/*` passes (or equivalent in Trusted Publishing flow)
- [ ] [HUMAN] If Docker image is being published: log in to Docker Hub and verify the expected tag is not already occupied
- [ ] [HUMAN] conda-forge feedstock PR opened (if applicable — see HUMAN_SIGNOFF_LIST.md)

---

## Publish Steps (in order)

> Complete ALL pre-release checks before starting these steps.

1. **Push the tag**: `git tag v{VERSION}` then `git push origin v{VERSION}`
2. **Watch `release.yml`**: monitor the GitHub Actions run until all jobs pass
3. **Verify PyPI**: `pip index versions zuntenium` should show the new version within ~5 min
4. **Verify Docker Hub**: `docker pull zuntenium/zuntenium:v{VERSION}` on an amd64 and arm64 machine (or via emulation)
5. **Verify GitHub Release**: check that the release page has the sdist, wheel, SHA256SUMS.txt, and SBOM attached
6. **Announce**: post to appropriate channels (Discord, Slack, newsletter, etc.)
7. **Update COMPATIBILITY.md**: mark the release date and update evidence timestamps

---

## Post-release
- [ ] Verify install from PyPI in a fresh virtual environment: `python -m pip install zuntenium=={VERSION}`
- [ ] Run smoke test on installed package: `python -m zuntenium doctor`
- [ ] Update `COMPATIBILITY.md` "Last reviewed" date
- [ ] Open next milestone on GitHub
- [ ] File any post-release issues found during publication in COMPAT_LEDGER.md

---

## Escalation / Blocker Protocol
- If any [CI] item fails: **stop the release**, file a COMPAT_LEDGER entry, fix, and restart from the beginning
- If any [HUMAN] item cannot be completed by the responsible person within 48 hours: escalate to maintainer
- Breaking changes vs. last release: **stop**, update CHANGELOG with migration instructions, restart

---

*This checklist is version-controlled. Changes require a PR and maintainer review.*

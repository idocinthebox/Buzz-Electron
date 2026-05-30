# Security — CVE Mitigation Report

This fork applies dependency security mitigations on top of upstream
[Buzz](https://github.com/chidiwilliams/buzz). Mitigations are **safe,
non-breaking version bumps only** — nothing that risks the
torch / faster-whisper / nemo CUDA stack. Breaking or unfixed advisories are
documented below and **deferred**, not silently shipped.

Audit tooling: `pip-audit` (Python deps) and `npm audit` (Electron deps).
Raw audit records are committed: `renamer-ui/scripts/pip_audit.json` (before)
and `renamer-ui/scripts/pip_audit_after.json` (after).

## Summary

| | Packages with advisories | Total advisories |
|---|---|---|
| **Before** | 22 | 76 |
| **After**  | 6  | 20 |
| **Cleared** | **16** | **56** |

`npm audit` (Electron `renamer-ui`): **0 vulnerabilities** (Electron 42.3.0).

## Mitigated (16 packages, 56 advisories cleared)

| Package | From → To | Advisories cleared |
|---|---|---|
| aiohttp        | 3.13.2 → 3.13.5 | 18 (CVE-2025-69223…69230, CVE-2026-22815, CVE-2026-34513…34525) |
| nltk           | 3.9.2 → 3.9.4   | 7 (CVE-2026-33230/33231, PYSEC-2026-96/98/99 …) |
| onnx           | 1.20.0 → 1.21.0 | 6 (CVE-2026-34445/34446/27489, PYSEC-2026-103/104 …) |
| pillow         | 12.0.0 → 12.2.0 | 6 (CVE-2026-25990/40192/42308/42309/42310/42311) |
| gitpython      | 3.1.45 → 3.1.50 | 4 (CVE-2026-42215/42284/44244 + GHSA-mv93-w799-cj2w) |
| urllib3        | 2.6.1 → 2.7.0   | 3 (CVE-2026-44432/44431/21441) |
| filelock       | 3.20.0 → 3.29.0 | 2 (CVE-2025-68146, CVE-2026-22701) |
| werkzeug       | 3.1.4 → 3.1.8   | 2 (CVE-2026-21860/27199) |
| idna           | 3.11 → 3.17     | 1 (CVE-2026-45409) |
| jaraco.context | 6.0.1 → 6.1.2   | 1 (CVE-2026-23949) |
| mako           | 1.3.10 → 1.3.12 | 1 (CVE-2026-44307) |
| marshmallow    | 3.26.1 → 3.26.2 | 1 (CVE-2025-68480) |
| protobuf       | 5.29.5 → 5.29.6 | 1 (CVE-2026-0994) |
| pygments       | 2.19.2 → 2.20.0 | 1 (CVE-2026-4539) |
| requests       | 2.32.5 → 2.34.2 | 1 (CVE-2026-25645) |
| virtualenv     | 20.35.4 → 20.39.1 | 1 (CVE-2026-22702) |

Constraints are pinned in `pyproject.toml` (direct deps inline; transitive deps
under the "Security floors" block).

## Known / Deferred (not mitigated)

Left unchanged to avoid breaking the ML stack or because no fix exists yet:

| Package | Advisories | Why deferred |
|---|---|---|
| transformers 4.53.3 | 9 | 8 have **no fixed release**; the only fix (CVE-2026-1839) is `5.0.0rc3`, a major prerelease that risks the Whisper pipeline. |
| nemo-toolkit 2.5.3 | 4 | Fix needs 2.6.x, which **drops the CUDA 12 pin** the project depends on. |
| pip 25.0.1 | 4 | Build/dev tool, not shipped in the packaged app. Upgrade to 26.x deferred. |
| pyarrow 22.0.0 | 1 | Fix is 23.0.1, a **major** bump (datasets dependency). |
| pytest 7.4.4 | 1 | Dev-only; fix needs pytest 9 (major). |
| pytorch-lightning 2.6.0 | 1 | **No fixed release** (CVE-2026-31221). |

## Re-running the audit

```bash
uv run pip-audit                 # or: .venv/Scripts/python -m pip_audit
cd renamer-ui && npm audit
```

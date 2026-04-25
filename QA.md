# QA Methodology

## Project Type

Python library/CLI — no web endpoints, no K8s deployment to verify browser flows against.
GUI server is local dev tool only (binds 0.0.0.0 intentionally).

## Applicable QA Steps

- Tests: `make test` — 410+ integration tests, 50%+ coverage, no mocks
- Lint: `make lint` — ruff (bandit security rules), mypy strict, isort, black
- Pre-commit: full suite including gitleaks, semgrep, vulture, detect-secrets
- Code review: manual diff review for prompt/config changes

## Not Applicable

- Browser QA / Playwright: no public web UI
- Schemathesis / ZAP / autoqa: no HTTP API
- K8s logs: deployment is simple container, no complex orchestration to debug
- SonarCloud: not integrated for this project

## Known Pre-existing Issues

- S104 (ruff/bandit) in web server/CLI — intentional bind to 0.0.0.0, suppressed with `# noqa: S104`
- pylint duplicate-code between `certamen` (legacy) and `certamen` — scoped to `certamen/` only in pre-commit
- pip-audit may flag pip itself — transient CVE, added to ignore list

## Effective Strategies

- Run `make test` first — fast feedback (410 tests in ~7s)
- Pre-commit hooks catch most issues before commit
- `treemapper . --diff` useful for reviewing prompt/config changes across many files
- Docker builds don't follow symlinks — always use real files for anything copied in Dockerfile
- JSONC files (tsconfig) with comments must be excluded from `check-json` hook
- pylint duplicate-code across legacy/core packages: scope to `certamen/` only, not `src/`
- markdownlint on generated reports/docs: exclude `benchmarks/reports/` and `docs/` dirs

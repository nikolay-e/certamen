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

- 4x S104 (ruff/bandit) in `src/certamen/web/server.py` and `src/certamen/cli/main.py` — intentional bind to 0.0.0.0 for local GUI server
- pylint duplicate-code between `certamen` (legacy) and `certamen_core` packages — known tech debt from merge, skipped in pre-commit
- pip-audit may flag pip itself — transient CVE, not actionable

## Effective Strategies

- Run `make test` first — fast feedback (410 tests in ~7s)
- Pre-commit hooks catch most issues before commit
- `treemapper . --diff` useful for reviewing prompt/config changes across many files

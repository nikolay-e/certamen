# QA Methodology — certamen

> Generic QA patterns (SonarCloud rules, pre-commit, ArgoCD, etc.) live in the global `/qa` skill. This file holds project-specific overrides only.

## Project Type

Python library/CLI — no web endpoints, no K8s deployment to verify browser flows against.
GUI server is local dev tool only (binds 0.0.0.0 intentionally).

## Applicability Matrix

**Applicable:**

- Tests: `make test` — 225+ integration tests, 30%+ coverage, no mocks
- Lint: `make lint` — ruff (bandit security rules), mypy strict, isort, black
- Pre-commit: full suite including gitleaks, semgrep, vulture, detect-secrets
- Code review: manual diff review for prompt/config changes

**Not Applicable:**

- Browser QA / Playwright: no public web UI
- Schemathesis / ZAP / autoqa: no HTTP API
- K8s logs: deployment is simple container, no complex orchestration
- SonarCloud project key is `nikolay-e_arbitrium-core` (old name); `sonar-project.properties` scopes analysis to `src/certamen/` only

## Project-Specific Findings

- **S104 (ruff/bandit)** in web server/CLI — intentional bind to 0.0.0.0, suppressed with `# noqa: S104`.
- **pylint duplicate-code** between `certamen` (legacy) and `certamen` — scoped to `certamen/` only in pre-commit; do NOT scope to `src/`.
- **pip-audit may flag pip itself** — transient CVE, added to ignore list.
- **SonarCloud BLOCKER on intentional bcrypt dummy hash**: add `# NOSONAR` + `# pragma: allowlist secret`.
- **JSONC files** (tsconfig with comments) must be excluded from `check-json` hook.
- **markdownlint** on generated reports/docs: exclude `benchmarks/reports/` and `docs/` dirs.
- **After package rename**, search BOTH `.github/workflows/` and `Makefile` for stale source paths — `--cov=src/<old_name>` causes "0.00% coverage" failure on CI while passing locally.
- **Docker builds don't follow symlinks** — always use real files for anything copied in Dockerfile.
- **CI Windows path test**: use `set.issubset(set(path.parts))` not `"src/certamen/workflows" in str(path)` to avoid backslash separator failure.

## Coverage Threshold

Web interface (`interfaces/web/`) and logging infrastructure (`shared/logging/`) contribute 0% coverage in CI as they require runtime; exclude them from `coverage.omit`. Threshold of 30% is correct for integration-test-only project.

## GUI / Workflow Editor

- GUI server requires `OLLAMA_BASE_URL` env to actually run workflows that include Ollama models; `CERTAMEN_SKIP_AUTH=true` for dev (skips bcrypt + DB init).
- Workflow Editor's WebSocket connects to `/ws` for live execution events; if backend uses lazy auth import path it must also bypass when `CERTAMEN_SKIP_AUTH=true`.

## Ollama Smoke Test

- Use `venv/bin/certamen` not system `certamen` (system binary may point to old package name).
- Always set `OLLAMA_BASE_URL=http://localhost:11434`.
- **qwen3 models with thinking mode**: use `gemma3:1b` as substitute in QA config — qwen3 fills `max_tokens` with `<think>...</think>` blocks leaving no room for actual answer.

## Workflow Node Patterns (project-specific)

- `tournament/rank` → `extract_insights.model` flow: rank node outputs model config as dict; `ExtractInsightsNode` must call `ensure_single_model_instance()` before `.generate()`.
- `flow/gate` → `synthesize.model` flow: gate returns champion as raw dict; `SynthesizeNode` must call `ensure_single_model_instance()` — same pattern as `ExtractInsightsNode`.
- **Pattern**: any workflow node accepting a `model` or `champion` input from `gate`/`rank` nodes must handle dict input via `ensure_single_model_instance()` before calling `safe_generate()`.
- **Executor termination**: `_is_iteration_done` must check ALL `node_outputs` entries (not just last layer tasks) — gate node in early layers was not detected otherwise.
- **Small models (1B–4B) as judges**: rankings will be empty (can't produce parseable `LLM1: X/10` scores); expected behavior, not a bug — tournament still terminates and produces synthesis.

## `asyncio.CancelledError` discipline

- In periodic loop: always `raise`, not `break`.
- In shutdown (task we explicitly cancelled): comment OK.
- At top level: use `finally` only.

## aiohttp handlers + SonarCloud S7503

- aiohttp's typed handlers require `async def` even when no `await` is needed (signature is `Callable[[Request], Awaitable[StreamResponse]]`). SonarCloud S7503 ("remove async, no awaits") is a wrong fix here — reverting to sync triggers mypy strict failures in `interfaces/web/` (3 in `runs.py`, 4 in `server.py`).
- `websocket_handler` returns either `WebSocketResponse` on success or `Response(status=403/503/429)` for pre-upgrade rejections. Annotate return as `web.StreamResponse` (parent of both) — not `WebSocketResponse`.
- Missing third-party stubs (`bcrypt`, `jwt`, `psycopg2`) live in mypy override `ignore_missing_imports = true` block in `pyproject.toml`.

## SonarCloud — project-specific

- Quality gate failed on `new_security_rating=3` because of S8565 ("missing lock file"). Fix: `uv lock` and commit `uv.lock` — Sonar accepts uv lock for Python projects, even though we ship as a wheel without one historically.
- Keychain `sonarcloud-token` is analysis-scope, not user-scope: `api/authentication/validate?token=…` returns valid but `api/users/current` shows `isLoggedIn: false` and any management POST (hotspot status change, issue resolution) returns 401. Marking hotspots SAFE requires manual UI action or a personal user token from 1Password.
- **Valid user-scope token in 1Password Sonarcloud LOGIN item is the `MCP TOKEN` concealed field** (not the `password` field, which is empty). Sign-in is via GitHub OAuth — there's no user password to extract. After `op item get vtej74af4ew6tdqnvinqno35zy --fields "MCP TOKEN" --reveal`, mirror to Keychain via `security add-generic-password -a "$USER" -s "sonarcloud-token" -w "$TOKEN" -U`. The older 1Password item `Sonar Cloud Token` (id qd7gbl7xpggjrsybhao5akzxxa) has been rotated and returns `valid:false`.
- Hotspots TO_REVIEW do NOT break the quality gate unless they appear in the `new_code` window — the metric is `new_security_hotspots_reviewed`. The 22 long-standing hotspots in this repo are safe-by-construction (non-crypto `random` for sim/shuffle; regex over bounded LLM output; `/tmp` literals in test fixtures) and can stay TO_REVIEW indefinitely without blocking CD.

## NOSONAR + black 79-char wrapping collision

- `pyproject.toml` enforces black `line-length = 79`. A signature already near the limit (e.g. `async def list_runs(_request: web.Request) -> web.Response:` = 60 chars) cannot fit `# NOSONAR(python:S7503)` (24 chars including spaces) inline — total 84+ chars, black wraps the whole signature, and the NOSONAR ends up on the closing `):` line, which Sonar IGNORES (per global skill: NOSONAR must be on `def` or `except` keyword line).
- Workaround: use bare `# NOSONAR` (suppresses ALL rules on the line; 11 chars), keep on the `def` line, and place a follow-up `# Sxxxx: reason` comment as the function body's first line for human readers. This is the pattern used in `interfaces/web/runs.py` and `interfaces/web/server.py` for the 6 aiohttp typed-handler S7503 suppressions.
- For typescript:S4325 false positives (Sonar thinks `as Record<string, ModelInfo>` is redundant from `Record<string, unknown>`, but `tsc -b` requires the assertion), cast through `unknown` first: `as unknown as Record<string, ModelInfo>` — TypeScript accepts; Sonar does not flag the double-cast pattern.

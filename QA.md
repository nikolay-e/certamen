# QA Methodology

## Project Type

Python library/CLI — no web endpoints, no K8s deployment to verify browser flows against.
GUI server is local dev tool only (binds 0.0.0.0 intentionally).

## Applicable QA Steps

- Tests: `make test` — 225+ integration tests, 30%+ coverage, no mocks
- Lint: `make lint` — ruff (bandit security rules), mypy strict, isort, black
- Pre-commit: full suite including gitleaks, semgrep, vulture, detect-secrets
- Code review: manual diff review for prompt/config changes

## Not Applicable

- Browser QA / Playwright: no public web UI
- Schemathesis / ZAP / autoqa: no HTTP API
- K8s logs: deployment is simple container, no complex orchestration to debug
- SonarCloud: project key is `nikolay-e_arbitrium-core` (old name); `sonar-project.properties` scopes analysis to `src/certamen/` only

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
- After package rename, search BOTH `.github/workflows/` and `Makefile` for stale source paths — `--cov=src/<old_name>` causes "0.00% coverage" failure on CI while passing locally
- GUI server requires `OLLAMA_BASE_URL` env to actually run workflows that include Ollama models; `CERTAMEN_SKIP_AUTH=true` for dev (skips bcrypt + DB init)
- Workflow Editor's WebSocket connects to `/ws` for live execution events; if backend uses lazy auth import path it must also bypass when `CERTAMEN_SKIP_AUTH=true`
- React error #31: rendering an object as a child — wrap any `data.error` in a string check before rendering (errors come from backend as `{type, message, node_id, node_type}` objects, not strings)
- Ollama smoke test: use `venv/bin/certamen` not system `certamen` (system binary may point to old package name); always set `OLLAMA_BASE_URL=http://localhost:11434`
- qwen3 models with thinking mode: use gemma3:1b as substitute in QA config — qwen3 fills max_tokens with `<think>...</think>` blocks leaving no room for actual answer
- CI Windows path test: use `set.issubset(set(path.parts))` not `"src/certamen/workflows" in str(path)` to avoid backslash separator failure
- Coverage threshold: web interface (interfaces/web/) and logging infrastructure (shared/logging/) contribute 0% coverage in CI as they require runtime; exclude them from coverage.omit; threshold of 30% is correct for integration-test-only project
- `tournament/rank` → `extract_insights.model` flow: the rank node outputs model config as dict; `ExtractInsightsNode` must call `ensure_single_model_instance()` before calling `.generate()`
- `flow/gate` → `synthesize.model` flow: gate returns champion as raw dict; `SynthesizeNode` must call `ensure_single_model_instance()` — same pattern as ExtractInsightsNode
- Pattern: any workflow node that accepts a `model` or `champion` input from `gate` or `rank` nodes must handle dict input via `ensure_single_model_instance()` before calling `safe_generate()`
- Executor termination: `_is_iteration_done` must check ALL `node_outputs` entries (not just last layer tasks) — gate node in early layers was not detected
- Small models (1B-4B) as judges: rankings will be empty because they can't produce scores in parseable format (LLM1: X/10); expected behavior, not a bug — tournament still terminates and produces synthesis
- SonarCloud project key is `nikolay-e_arbitrium-core` (old name); `sonar-project.properties` scopes analysis to `src/certamen/` only
- SonarCloud BLOCKER on intentional bcrypt dummy hash: add `# NOSONAR` + `# pragma: allowlist secret`
- `pull_request_target` with `github.triggering_actor` is forgeable — use `pull_request` + `github.actor`
- `asyncio.CancelledError` in periodic loop: always `raise`, not `break`; in shutdown (task we explicitly cancelled): comment OK; at top level: use `finally` only
- SonarCloud NOSONAR in Python: must be on the `except` keyword line or function `def` line — NOT on the closing `):`  of a multi-line except. Black reformats `except X:  # NOSONAR` back to multi-line if comment is too long; use short comment `# NOSONAR` to stay under 88 chars and prevent Black expansion
- SonarCloud S6551 (unknown type in template literals / String coercion): extract to typed variable first (`const x = val as string;`) — inline `as string` inside JSX braces is rejected by `tsc -b` even though `tsc --noEmit` passes; always run `npm run build` (not just `npm run type-check`) to catch these
- SonarCloud S7735 (negated ternary): extract nested loop logic to a named helper function — eliminates both S7735 (flip ternary) and S3776 (complexity) in one shot
- SonarCloud S6819 (interactive handler on non-interactive element): convert drag-handle `<div>` to `<button type="button" draggable>` — buttons support the `draggable` attribute; add CSS resets (`background: none; border: none; width: 100%; text-align: left; font-family: inherit`)
- `tsc -b` vs `tsc --noEmit`: `tsc -b` (used in `npm run build`) is stricter — run `npm run build` before committing, not just `npm run type-check`; `!!value` coerces unknown to boolean for JSX short-circuit; typed variable extraction is safer than inline `as` casts in JSX expressions

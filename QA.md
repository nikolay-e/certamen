# QA Methodology — certamen

> Generic QA patterns (pre-commit, ArgoCD, etc.) live in the global `/qa` skill. This file holds project-specific overrides only.

## Project Type

Python library/CLI — no web endpoints, no K8s deployment to verify browser flows against.
GUI server is local dev tool only (binds 0.0.0.0 intentionally).

## Applicability Matrix

**Applicable:**

- Tests: `make test` — 225+ integration tests, 30%+ coverage, no mocks
- Lint: `make lint` — ruff (lint + format + import sort + bandit security rules), pyright strict
- Pre-commit: full suite including gitleaks, semgrep, vulture, detect-secrets
- Code review: manual diff review for prompt/config changes
- **Ollama smoke**: `make discover-ollama` + at least one slim-config run via `certamen --config /tmp/cert-*.yml` to catch provider-prefix and slim-schema regressions
- **Workflow integration**: `certamen workflow validate` + `certamen workflow execute` over EVERY YAML under `examples/workflows/` AND `src/certamen/workflows/` — validate-only is insufficient (catches schema typos but not runtime issues like wrong property names, missing model providers, empty outputs)
- **GUI QA (Playwright MCP)**: build `cd frontend && npm run build && npm run lint`, start `certamen gui --port 8765` with `CERTAMEN_SKIP_AUTH=true`, navigate to `http://localhost:8765/`, exercise Workflow Editor + Tournament Results tabs, watch console errors and `/api/runs` content (question/champion render)

**Not Applicable:**

- Schemathesis / ZAP / autoqa: no HTTP API surface intended for external traffic (the GUI server is dev-only, binds 0.0.0.0 intentionally, has no OpenAPI). The crawler / accessibility checks are out-of-scope until/unless we ship the GUI as a hosted product.
- K8s logs: deployment is simple container, no complex orchestration
- SonarCloud: removed 2026-05 (`sonar-project.properties` deleted). Coverage via pytest-cov, security via ruff `S*` + bandit + CodeQL, secrets via gitleaks/detect-secrets — Sonar duplicated all of it

## Project-Specific Findings

- **S104 (ruff/bandit)** in web server/CLI — intentional bind to 0.0.0.0, suppressed with `# noqa: S104`.
- **pylint duplicate-code** between `certamen` (legacy) and `certamen` — scoped to `certamen/` only in pre-commit; do NOT scope to `src/`.
- **pip-audit may flag pip itself** — transient CVE, added to ignore list.
- **Intentional bcrypt dummy hash** (timing-attack prevention) needs `# pragma: allowlist secret` for detect-secrets.
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
- **Run every example workflow at least once per `/qa` pass**: `for wf in examples/workflows/*.yml src/certamen/workflows/*.yml; do certamen workflow execute "$wf"; done`. Validate-only ran green for both `multi-model-comparison.yml` and `prompt-template.yml` while runtime silently produced empty outputs — only end-to-end execution surfaced the broken `pages:`/`mode:` properties.
- **`diamond-tournament` requires 4 LLM nodes**: slim config must declare exactly N models where N matches the workflow's `simple/llm` node count. Mismatch errors loudly (`expects N models, got M`) — good. For `diamond-tournament` use 4 entries; reuse the same model under different keys (`gemma3_1b_a`/`gemma3_1b_b`) when only 1–2 local models are available.
- **Per `slim.py` schema**: extra top-level keys are forbidden. `judges:` does NOT exist; judges are defined inside the workflow's tournament/judge node. The slim config holds ONLY `workflow`, `question`, `models`, `overrides`, `secrets`, `outputs_dir`, `logging` (see `SlimConfig` in `src/certamen/infrastructure/config/slim.py`).

## Workflow YAML Authoring Traps

- **`simple/text` node**: properties are `texts: [str, ...]`, `separator: str`, and (internal) `pages`, `current_page`, `hidden`. **Putting seed input into `pages:` instead of `texts:` produces silent empty outputs** — `TextNode.execute()` only reads `pages` when `input_text` is connected. Symptom: workflow validates AND executes successfully, all LLM nodes get empty prompts, all `simple/text` output nodes show `output_text: ""`. Seen on `examples/workflows/multi-model-comparison.yml` and `examples/workflows/prompt-template.yml`. Fix: convert `pages: [["seed text"]]` + `mode: append` → `texts: ["seed text"]` + `separator: "\n"`. The `mode:` property does NOT exist on `simple/text` (only on `flow/gate`).
- **`simple/llm` model_name MUST include provider prefix** when provider is `ollama`: `model_name: ollama/gemma3:1b`, not `model_name: gemma3:1b`. Without the prefix, LiteLLM raises `BadRequestError: LLM Provider NOT provided` and the model returns an error response. As of the 2026-05-17 fix, `LiteLLMModel._validate_required_fields` auto-prepends the prefix for `ollama` provider — but example workflows should still write the prefix explicitly for clarity.

## Ollama Provider Prefix + Error Classifier Foot-Gun

- `scripts/discover_ollama_models.py` (the `make discover-ollama` command) was generating slim configs with model_name missing the `ollama/` prefix AND with legacy `tournament:`/`knowledge_bank:` keys that `SlimConfig` rejects. Now fixed — generated config matches slim schema and uses `ollama/<name>` everywhere.
- `ExceptionClassifier` substring patterns must be long enough or word-delimited to avoid collisions with docs URLs and unrelated words. The pattern `"tps"` in `rate_limit` matched inside `https://` and misclassified EVERY `BadRequestError` (which contains a `https://docs.litellm.ai/...` link) as a rate-limit error → caller would retry forever for a permanent config bug. Wrapped short patterns (`tps`, `rpm`, `429`, `code: 6`) with surrounding spaces; added explicit `BadRequestError` mapping to `bad_request` (non-retryable) in `_EXCEPTION_TYPE_MAP`. When adding error-pattern strings: anything under ~5 chars MUST be space-delimited or word-bounded.
- Logged errors should include the actual error string, not the wrapper object. `ModelResponse` had no `__repr__`, so `logger.warning("... %s", response)` printed `<ModelResponse object at 0x...>`. Now logs `model=<name> error_type=<type> error=<text>` AND `ModelResponse.__repr__` returns the same shape for defensive logging.

## Tournament Event Metadata

- `tournament_started` MUST include `question` and `models` in payload — the Tournament Results sidebar in the GUI reads them via `_apply_event_to_run_summary` in `interfaces/web/runs.py`. Without these, the sidebar shows `(no question)` and the run is unidentifiable in a list of similar runs.
- `tournament_ended` MUST include `champion: {name, model_name, provider}` — extracted from output nodes whose payload contains `{"champion": {"model": {...}, "response": "...", "rankings": [...]}}` (matches the `tournament/champion` node schema). The summary stores the friendly `name`, falling back to `model_name`.
- Both fields are emitted from `_execute_workflow_dict` in `interfaces/cli/main.py` via `_extract_workflow_question` / `_extract_workflow_model_keys` / `_extract_champion_from_outputs`. Any future workflow that wants its run to be identifiable in the Runs sidebar must use a `simple/text` node with `id: "question"` (the slim_loader convention) and any `simple/llm` nodes' `properties.name` will appear as the model labels.

## GUI / Workflow Editor QA

- After any frontend change, run `cd frontend && npm run build && npm run lint` BEFORE manually testing. `tsc -b` (inside `npm run build`) is stricter than `tsc --noEmit` and catches type errors that `npm run lint` misses.
- The shipped `frontend/index.html` MUST include a `<link rel="icon" ...>` — absence causes a console-visible `404 /favicon.ico` on every page load. Use inline SVG data URL (e.g., `🏆` glyph) to avoid an asset round-trip.
- Health endpoint is `/health` NOT `/api/health` (see `_setup_routes` in `server.py:248`). Other routes ARE under `/api/`: `/api/models`, `/api/nodes`, `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/events`, plus `/api/runs/{id}/attach` (websocket).
- `CERTAMEN_SKIP_AUTH=true` is REQUIRED for local QA — without it, the GUI bootstrap tries to load `bcrypt`/JWT modules and may fail on env without `JWT_SECRET_KEY` / DB.
- `OLLAMA_BASE_URL=http://localhost:11434` is also needed for the GUI to populate the model dropdown via `/api/models`; otherwise the editor shows a barebones model list (LiteLLM static fallback only).

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

## aiohttp typed handlers + pyright

- aiohttp's typed handlers require `async def` even when no `await` is needed (signature is `Callable[[Request], Awaitable[StreamResponse]]`). Reverting to sync to satisfy a "no awaits" lint triggers strict type failures in `interfaces/web/` (3 in `runs.py`, 4 in `server.py`).
- `websocket_handler` returns either `WebSocketResponse` on success or `Response(status=403/503/429)` for pre-upgrade rejections. Annotate return as `web.StreamResponse` (parent of both) — not `WebSocketResponse`.
- Web/auth runtime deps (`aiohttp`, `bcrypt`, `PyJWT`, `psycopg2`) live in the `gui` optional-extra (`pip install certamen-core[gui]` / `uv sync --extra gui`). pyright resolves `psycopg2` via typeshed stubs; `bcrypt`/`jwt` carry inline `# pyright: ignore[reportMissingImports]` because the default dev env (`uv sync --extra dev`) does not install them.

## pre-commit ruff rev vs installed ruff

- `.pre-commit-config.yaml` `astral-sh/ruff-pre-commit` rev must stay aligned with `pyproject.toml`'s ruff constraint (currently `>=0.15.10,<1.0`). `ruff format` output can differ between minor versions, so a stale pre-commit rev makes CI pre-commit and local `make lint` (venv ruff) disagree — `make lint` rejecting files that pre-commit accepts, or vice versa. Bump the `ruff-pre-commit` rev whenever the pyproject ruff pin is upgraded. (Ruff now replaces black + isort + pyupgrade, so there is a single formatter/linter version to keep in sync instead of three.)

# QA Methodology — certamen

> Generic QA patterns live in the `/qa` skill. This file holds project-specific
> overrides only. Packaging QA (wheel/clean-venv E2E, `uv lock --check` hook,
> pyright-outside-venv, import-smoke, CI-Windows-path-test) is in the skill's
> **Packaging QA** section — referenced, not duplicated, below.

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
- SonarCloud: removed (`sonar-project.properties` deleted). Coverage via pytest-cov, security via ruff `S*` + bandit + CodeQL, secrets via gitleaks/detect-secrets — Sonar duplicated all of it, so it was retired rather than maintained twice

## Project-Specific Findings

- **S104 (ruff/bandit)** in web server/CLI — intentional bind to 0.0.0.0, suppressed with `# noqa: S104`.
- **pip-audit audits its OWN hook environment**, not certamen's deps (the venv/uv.lock). CVEs in pip-audit's transitive deps (e.g. `pip` itself, or `msgpack` via `cachecontrol`) surface here even though they're absent from certamen's supply chain — add the GHSA to the hook's `--ignore-vuln` list in `.pre-commit-config.yaml`. Do NOT `uv lock --upgrade-package <pkg>` for these (the package isn't in uv.lock, so it triggers a full-lock re-resolution churn instead).
- **Intentional bcrypt dummy hash** (timing-attack prevention) needs `# pragma: allowlist secret` for detect-secrets.
- **JSONC files** (tsconfig with comments) must be excluded from `check-json` hook.
- **markdownlint** on generated reports/docs: exclude `benchmarks/reports/` and `docs/` dirs.
- **After package rename**, search BOTH `.github/workflows/` and `Makefile` for stale source paths — `--cov=src/<old_name>` causes "0.00% coverage" failure on CI while passing locally.
- **Docker builds don't follow symlinks** — always use real files for anything copied in Dockerfile.

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
- **`diamond-tournament.yml` is the heavy full-pipeline E2E and runs ~15+ min** when all 4 models genuinely generate (iterative diverge→converge, `gate max_rounds: 6`, peer-review + interrogate + synthesis + knowledge-map). Its shipped `Model C: ollama/qwen3:4b` thinking-modes to an empty response → emits a non-fatal `WARNING [base] Model generation returned error response` and Model C contributes nothing; the tournament still completes and picks a champion (graceful). qwen3's empty short-circuit is *why* it appears to finish quickly. Swapping Model C to a generating model surfaces a second trap: weak 3B judges (orca-mini) return `apology/refusal instead of evaluation` (logged ERROR, scoring continues). **For a FAST full-pipeline E2E check use `tournament-elimination.yml`** (completes with a champion in ~1 min); reserve `diamond-tournament.yml` for when you can wait. Changing the shipped Model C roster is a curation decision, left to the maintainer.
- **Per `slim.py` schema**: extra top-level keys are forbidden. `judges:` does NOT exist; judges are defined inside the workflow's tournament/judge node. The slim config holds ONLY `workflow`, `question`, `models`, `overrides`, `price_overrides`, `secrets`, `outputs_dir`, `logging` (see `SlimConfig` in `src/certamen/infrastructure/config/slim.py`). `price_overrides` is a map of `model_name → {input_per_1m, output_per_1m}` consumed only by `certamen cost`.

## Cost Estimation & Tournament Pipeline

- **`certamen cost --config X.yml`** prints a bounded (`min/expected/max`) per-model, per-stage estimate WITHOUT running the tournament. Prices come from `litellm.model_cost` (+ config `price_overrides`, fail-loud on unpriced, `ollama`→$0). Reuses `load_and_materialize`, so it also catches slim-config errors (e.g. model-count vs `simple/llm`-node mismatch) for free.
- **The executor re-runs the whole forward DAG every gate iteration** (`async_executor.py` `while True`, no memoization; the CLI path builds models without a response cache). So divergence (`generate`/`interrogate`/`diverge_improve`) re-fires each round: a healthy 4-model diamond logs ~144 LLM calls over 4 iterations, not ~36. Any cost/call-count reasoning must account for this ×(iterations) multiplier.
- **Deterministic $0 pipeline E2E**: `tests/integration/test_e2e_tournament_pipeline.py` monkeypatches `LiteLLMModel.from_config` to return a stub `BaseModel` (parseable scores for peer_review, numbered questions for interrogation) and drives the real `AsyncExecutor` over the materialized diamond graph — asserts interrogation insights flow forward and synthesis/knowledge_map are non-empty, no API keys, no ollama. Use this pattern to verify graph/wiring changes without burning tokens or waiting ~15 min on ollama.
- **`diamond-tournament.yml` now runs `interrogate.rounds: 3`** (relentless multi-round follow-up interrogation) — this multiplies interrogation calls ~3× and lengthens real runs. Interrogation output flows via a new `insights` STRING output on `tournament/interrogate` into `diverge_improve` (the old edge asked for a non-existent handle and silently dropped it). Finalization (`synthesize`/`knowledge_map`) reads `diverge_improve.improved` (always populated) rather than `converge_improve.improved` (empty on the terminal iteration); `knowledge_map` reuses `champion` as its judge model (a dedicated `gate→judge` edge trips cycle detection).

## Workflow YAML Authoring Traps

- **`simple/text` node**: properties are `texts: [str, ...]`, `separator: str`, and (internal) `pages`, `current_page`, `hidden`. **Putting seed input into `pages:` instead of `texts:` produces silent empty outputs** — `TextNode.execute()` only reads `pages` when `input_text` is connected. Symptom: workflow validates AND executes successfully, all LLM nodes get empty prompts, all `simple/text` output nodes show `output_text: ""`. Seen on `examples/workflows/multi-model-comparison.yml` and `examples/workflows/prompt-template.yml`. Fix: convert `pages: [["seed text"]]` + `mode: append` → `texts: ["seed text"]` + `separator: "\n"`. The `mode:` property does NOT exist on `simple/text` (only on `flow/gate`).
- **`simple/llm` model_name MUST include provider prefix** when provider is `ollama`: `model_name: ollama/gemma3:1b`, not `model_name: gemma3:1b`. Without the prefix, LiteLLM raises `BadRequestError: LLM Provider NOT provided` and the model returns an error response. `LiteLLMModel._validate_required_fields` now auto-prepends the prefix for `ollama` provider — but example workflows should still write the prefix explicitly for clarity.

## Error Classifier Foot-Guns

- `scripts/discover_ollama_models.py` (the `make discover-ollama` command) once generated slim configs with model_name missing the `ollama/` prefix AND with legacy `tournament:`/`knowledge_bank:` keys that `SlimConfig` rejects. Now fixed — generated config matches slim schema and uses `ollama/<name>` everywhere.
- **`ExceptionClassifier` short-substring collision**: a substring error-pattern (e.g. `"tps"`) matched inside `https://` in every `BadRequestError` (whose message carries a `docs.litellm.ai` URL) → misclassified permanent config bugs as retryable rate-limits → infinite retry. When adding error-pattern strings: anything under ~5 chars MUST be space-delimited or word-bounded; `BadRequestError` is mapped to `bad_request` (non-retryable) in `_EXCEPTION_TYPE_MAP`.
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
- **`simple/llm` `PROPERTIES` schema must declare every key `_build_model_config` reads.** `system_prompt` and `web_search_options` are read from `node_properties` and applied, but were absent from `LLMNode.PROPERTIES` → `BaseNode._validate_properties` logged `Unknown properties {'system_prompt'}` on EVERY workflow using personas (knowledge-bank, tournament-elimination, chain-of-thought, diamond-tournament), even though the property worked at runtime (the validator only warns, never strips). The schema is also what the frontend editor renders from `/api/nodes`, so an undeclared-but-used property is invisible in the UI. Rule: any property the node's `execute`/`_build_model_config` consumes MUST be in `PROPERTIES`.

## `asyncio.CancelledError` discipline

- In periodic loop: always `raise`, not `break`.
- In shutdown (task we explicitly cancelled): comment OK.
- At top level: use `finally` only.

## aiohttp typed handlers + pyright

- aiohttp's typed handlers require `async def` even when no `await` is needed (signature is `Callable[[Request], Awaitable[StreamResponse]]`). Reverting to sync to satisfy a "no awaits" lint triggers strict type failures in `interfaces/web/` (3 in `runs.py`, 4 in `server.py`).
- `websocket_handler` returns either `WebSocketResponse` on success or `Response(status=403/503/429)` for pre-upgrade rejections. Annotate return as `web.StreamResponse` (parent of both) — not `WebSocketResponse`.
- Web/auth runtime deps (`aiohttp`, `bcrypt`, `PyJWT`, `psycopg2`) live in the `gui` optional-extra (`pip install certamen-core[gui]` / `uv sync --extra gui`). pyright resolves `psycopg2` via typeshed stubs; `bcrypt`/`jwt` carry inline `# pyright: ignore[reportMissingImports]` because the default dev env (`uv sync --extra dev`) does not install them.

## Frontend tooling (biome / deptry config drift)

- **`biome migrate --write` mis-converts `linter.rules.recommended: true` →
  `preset: "none"`** on biome 2.5.0 — i.e. it DISABLES every rule (the opposite
  of intent), while `biome ci` still exits 0, so the regression is invisible.
  The correct value is `preset: "recommended"`. After a biome bump, hand-fix the
  config: bump `biome.json` `$schema` to the installed version AND set
  `rules.preset` to `"recommended"` yourself — never trust `migrate`'s output.
  Reproduces with and without a `linter.domains` block. Filed upstream:
  biomejs/biome#10716.
- **deptry config**: `pep621_dev_dependency_groups` is deprecated (deptry ≥0.25)
  → use `optional_dependencies_dev_groups`. Surfaces as a non-fatal warning in
  `make lint`/CI; bump the key in `[tool.deptry]` when seen.

## pre-commit ruff rev vs installed ruff

- `.pre-commit-config.yaml` `astral-sh/ruff-pre-commit` rev must stay aligned with `pyproject.toml`'s ruff constraint (currently `>=0.15.10,<1.0`). `ruff format` output can differ between minor versions, so a stale pre-commit rev makes CI pre-commit and local `make lint` (venv ruff) disagree. Bump the `ruff-pre-commit` rev whenever the pyproject ruff pin is upgraded. (Ruff replaces black + isort + pyupgrade, so there is a single formatter/linter version to keep in sync instead of three.)

## CI / branch-protection / dependabot gotchas

- **Branch-protection required checks MUST mirror the CI job matrix.** Moving the
  macOS/Windows test matrix out of CI (`ci.yml`) into release-only (`cd.yml`
  `pre-release-tests`) left `Test (Python 3.12 / macos-latest)` and
  `Test (Python 3.12 / windows-latest)` as *required* contexts that CI no longer
  reports — every PR then sits in `BLOCKED`/"Expected, waiting for status"
  forever. Fix with `gh api --method PATCH repos/<o>/<r>/branches/main/protection/required_status_checks`
  passing only the contexts CI actually emits.
- **Dependabot must use the `uv` ecosystem, not `pip`.** This project is
  uv-managed (uv.lock + the `uv-lock` pre-commit gate). The `pip` ecosystem edits
  pyproject.toml constraints but never updates uv.lock, so every Python dep PR is
  born `BLOCKED` on the uv-lock check. The `uv` ecosystem updates pyproject +
  uv.lock atomically. Frontend deps need a separate `npm` ecosystem entry
  (`directory: /frontend`) — without it the frontend only receives repo-level
  *security* PRs, not version bumps.
- **`diff-context.yml` referenced a nonexistent `nikolay-e/treemapper-action`
  reusable workflow** → 0s "workflow file issue" failure on every PR. There is no
  published diff-context GitHub Action; the diff-context review is a local CLI
  step (`diffctx . --diff …`). The workflow was removed.
- **pre-commit pyright `additional_dependencies` omits the gui/auth runtime deps**
  (`psycopg2-binary`, `bcrypt`, `PyJWT`). So CI's pre-commit pyright cannot
  type-check `interfaces/web/auth/*` and stays green while `make lint` (run with
  `.[dev,gui]` installed in the venv) surfaces strict errors there (e.g.
  `reportUnnecessaryComparison` on a defensive `getconn() is None` guard). Run
  `make lint` with the gui extra installed to catch what CI misses; suppress
  genuinely-defensive-but-stub-dead guards with a targeted
  `# pyright: ignore[reportUnnecessaryComparison]`.

## CI gates & tooling

- **Type check is pyright (strict), not mypy.** `Unknown*` reports are disabled for untyped libs (litellm/sklearn) to mirror the old `ignore_missing_imports`; `pythonVersion = "3.11"` matches `requires-python`. See `/qa` skill: Packaging QA — bare `pyright` outside the venv resolves the wrong interpreter and floods false errors; run via `make lint` inside the venv.
- **`uv lock --check` is a pre-commit hook** (`astral-sh/uv-pre-commit`, id `uv-lock`) — see `/qa` skill: Packaging QA. Any dependency edit MUST be followed by `uv lock` or this gate fails.
- **deptry guards dependency completeness** (CI `test` job + `make lint`). Config: `dev` is a dev-group; `PyJWT→jwt` / `psycopg2-binary→psycopg2` name-map; `tiktoken`/`aiofiles`/`google-auth` are `DEP002`-ignored (litellm runtime extras we never import directly).
- **import-smoke test** (`tests/integration/test_import_smoke.py`) walks every module — needs the `gui` extra installed AND sets `CERTAMEN_JWT_SECRET` (the auth package hard-fails at import without a ≥32-char secret). CI `test` job installs `.[dev,gui]`. See `/qa` skill: Packaging QA (import smoke).
- **CI Windows path test**: use `set.issubset(set(path.parts))` not `"src/certamen/workflows" in str(path)` to avoid backslash separator failure (see `/qa` skill: Packaging QA).
- **Frontend is gated in CI** (`Frontend` job: `biome ci` + `tsc -b` + `vitest run`). Vitest tests exercise real behavior (real store via `getState`, real hooks via `renderHook`) — no mocks.
- **codespell skips lockfiles** (`package-lock.json`, `uv.lock`) — hash fragments trip false positives.
- **A standalone `.importlinter` file takes priority over `pyproject.toml`'s `[tool.importlinter]` section.** A legacy `.importlinter` (ini-format, from the certamen-framework merge) referenced modules that no longer exist (`certamen.config`, `certamen.models`, `certamen.certamen`), so `lint-imports` crashed with `Module 'certamen.config' does not exist` on every run — silently, because CI's `architecture-checks` job is `continue-on-error: true` and swallows any failure into `::warning::Architecture violations detected`. This meant the correct, current layer-boundary contracts already defined in `pyproject.toml` (domain/ports/shared independence, Clean Architecture layering) were **never actually enforced**, for as long as the stale file existed. `lint-imports --verbose` names the exact contract/module it chokes on — check for a stray `.importlinter` file whenever that error references a module that doesn't exist in `src/`. Deleted; contracts now run for real (4 kept, 0 broken).
- **After any frontend dependency bump that includes `@biomejs/biome`, `frontend/biome.json`'s `$schema` URL must be bumped to match** — otherwise `biome ci` in CI (which installs the new version via `npm ci`) emits "configuration schema version does not match the CLI version" every run. Not a hard failure, but a recurring warning; bump the schema URL as part of the same commit as the dependency bump.

## CLI output bugs (found via manual `certamen` smoke testing)

- **`Display.print()` (`interfaces/cli/ui.py`) force-encoded all colored CLI output to ASCII** (`text.encode("ascii", errors="replace")`), turning every non-ASCII character — arrows in workflow names/descriptions (`Diverge → Converge`), emoji in LLM responses, non-English names — into a literal `?`. Reproduces even on a UTF-8 terminal/locale; verify with `certamen workflow show <name> | xxd` and grep for `3f` where a multi-byte UTF-8 sequence should be. Fix: encode with `sys.stdout.encoding` instead of hardcoded `"ascii"`. Only `cli_cyan`/`cli_error`/`cli_success`/`cli_warning`/`cli_info` (all routed through `Display.print`) were affected; raw `print()` call sites (e.g. workflow execute's output dump) were already fine.
- **Wiring a `gate`/`eliminate` champion (raw model config dict) into a `simple/text` output node dumps every internal field** (`system_prompt`, `context_window`, `base_url`, `temperature`, ...) instead of showing the model name — same bug class as the previously-fixed champion-stringification findings (`shared.mapping_utils.model_display_name`), but in `TextNode._to_page`'s dict-formatting branch, which the earlier fix pass didn't touch. Reproduces by running `examples/workflows/tournament-elimination.yml` end-to-end and inspecting `champion_output`. Fix: when the dict has both `model_name` and `provider` keys (the model-config shape), render via `model_display_name()`; otherwise keep the existing key/value dump (needed for generic debugging output).
- **`certamen workflow validate` never checked node types against the registry** — only `WorkflowLoader.load_from_file`'s YAML-schema check ran, so a workflow with a typo'd/unknown `type:` on a node reported "Workflow is valid" (exit 0) and only failed later, with a different and less discoverable error, at `workflow execute`. Defeats the entire purpose of a pre-flight validate command. Fix: `_validate_workflow_file` now cross-checks every node's `type` against `certamen.application.workflow.registry.registry.get(...)` and fails validation with the offending type name(s) if any are unregistered.
- **`scripts/discover_ollama_models.py` used Ollama's `/api/tags` `details.family` field (e.g. `"llama"`) as `display_name`**, not the actual model tag — since many unrelated models share an architecture family (Yi, Llama, Mistral-derived models all commonly report `"llama"`), generated configs had duplicate `display_name`s across genuinely different models (verify: `grep display_name: config.yml | sort | uniq -c | sort -rn`), breaking champion/report disambiguation for anyone using `make discover-ollama` with more than a couple of local models. Fix: use the actual `model["name"]` (the Ollama tag) instead.

## Workflow executor — cycles vs feedback loops

- A **back-edge in a workflow graph is NOT rejected as a cycle** — the executor classifies it as a bounded *feedback loop* (the mechanism behind tournament gate-loops) and runs the iteration loop up to `max_iterations` (default 20), then stops with normal `outputs`. `GraphValidationError("Graph contains a cycle")` only fires for a cycle *within a single execution layer* (non-feedback). So the safety property to assert for a cyclic graph is **termination** (returns `outputs`, no hang), not an error. See `tests/integration/test_executor_gaps.py`.
- `simple/text` reads `texts:` only when no `input_text` is connected; seed text placed in `pages:` is silently ignored → empty `output_text` (validates + "runs" green). Covered by `test_executor_gaps.py`.

## GUI runtime bugs (found via Playwright MCP)

- **Tournament-results WS reconnect storm**: `TournamentView`'s attach-WebSocket `useEffect` must depend ONLY on `selectedId`. Including `liveStatus` (which the effect itself sets to connecting→live) re-runs the effect on every status change → close+reopen WS forever (192+ `WebSocket closed before connection established` warnings on opening any run). The `// eslint-disable react-hooks/exhaustive-deps` masked it. Use a functional update in `onclose` (`setLiveStatus(prev => prev === "live" ? "ended" : prev)`) instead of reading `liveStatus`.
- **React error #31 on champion**: `tournament_ended.payload.champion` is `{name, model_name, provider}` (object) for current runs, but older runs emit a string. Rendering it directly crashes the run-detail subtree. Extract a display string (`typeof === "string" ? it : c.name ?? c.model_name`) before assigning to a rendered field. (Reconnect storm was masking this until the WS stabilised.)
- **Editor fires ~1 `POST /api/validate` per node on load** (28 for diamond-tournament) — all 200 but a chatty re-validate; debounce opportunity, not a bug.
- Local `certamen gui` shows version `v0.0.0-placeholder` (the `GIT_SHA` sed runs only at Docker build) — expected, not a finding.
- **Completed-run detail stuck on "● LIVE" + 10-min idle WS.** Opening a finished run: `attach_run_websocket` replays all buffered events (incl. `tournament_ended`) as `{type:"event"}` but `_replay_events` never sent `{type:"ended"}`; `_tail_events` only sends "ended" for `tournament_ended` in *new* events, so for an already-complete run it idled `idle_timeout=600s` before closing → status frozen at "live" and a 10-min hung WS per opened run. Fix: `_replay_events` returns an `already_ended` flag; the handler sends `{type:"ended"}` and closes immediately when the replay already contained `tournament_ended`.
- **PhaseStrip showed "Waiting for tournament to start…" forever on workflow runs.** The strip is built on `phase_started`/`phase_completed` events, but the workflow executor emits node-level events (`node_start`/`node_complete`/`iteration_start`), never phase events — so `allPhases` is always empty. Fix: when a `tournament_started` event exists but no phase events, render nothing instead of the stale "waiting to start" message.

---

Generic QA patterns live in the `/qa` skill — do not duplicate here.

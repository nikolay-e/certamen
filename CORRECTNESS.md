# Correctness Audit Log — certamen

Append-only. Each run layers on top; do not rewrite prior entries.

Severity: 🔴 wrong in normal use · 🟡 wrong in edge case · 🔵 misleading name/comment

---

## 2026-06-20 — commit b708e09

Method: 5 parallel subsystem scouts → 2 adversarial verifiers (REFUTE-by-default) →
synthesis. Deterministic layer (ruff, pyright strict, deptry, frontend `tsc -b` +
biome, full test suite 243 passed) already green, so existence-class issues
(non-existent APIs, wrong signatures, fake packages) are ruled out — findings below
are logic/behavior only.

### 🔴 Confirmed

**C1. `certamen workflow execute <file>` never terminates a tournament early — always
runs the hardcoded 20 iterations.**
`src/certamen/application/execution/executor.py:717` (`_check_termination`)

```python
for node_id, _ in tasks:          # tasks == last_tasks == FINAL layer only
    if node_id in node_outputs and node_outputs[node_id].get("done") is True:
```

The `done: True` termination signal is emitted by the `flow/gate` node
(`flow.py:381`), which sits mid-graph; the final execution layer is the terminal
text/output nodes. So `gate.done` is never in `tasks`, the check never fires, and
`BaseExecutor._run_execution_loop` runs to `max_iterations = 20` (hardcoded,
`executor.py:750`). The sibling `AsyncExecutor._is_iteration_done`
(`async_executor.py:390`) scans **all** `node_outputs` and is correct — the two
executors disagree. The CLI `certamen workflow execute <file>` path uses the buggy
SyncExecutor (`interfaces/cli/main.py:287`); `certamen --config` (slim) uses the
correct async path.
Runtime corroboration: this QA pass observed `tournament-elimination.yml` finishing
with `iterations=20` (the cap, not an early champion stop), and
`diamond-tournament.yml` running past 20 min — both via `workflow execute`. Effect:
~20× wasted compute and no early-champion stop on the documented `workflow execute`
command. Fix: make `_check_termination` scan all `node_outputs` (mirror the async
executor), and consider making `max_iterations` configurable.

### 🟡 Confirmed

**C2. EliminateNode eliminates by dict insertion order on tied/missing scores.**
`src/certamen/application/workflow/nodes/flow.py:285-286` via
`rank_by_scores` (`nodes/base.py:195-197` `sorted(..., key=lambda k: scores.get(k,0))`).
`sorted` is stable, so equal/missing scores preserve insertion order and a model
absent from `scores` is treated as 0 (eliminated first). `extract_scores_from_evaluation`
returns `{}` on any incomplete evaluation (`scoring.py:74`), and weak 1B–4B judges
routinely produce unparsable scores (documented in QA.md). So in the common
small-model case elimination is arbitrary insertion-order, not worst-performer.

**C3. `normalize_score` is dead code — scores enter aggregation unclamped/unnormalized.**
`src/certamen/domain/tournament/scoring.py:78` defines `normalize_score`
(reject <0.5 / >10.5, normalize 0–1 → 0–10) but no `src/` caller exists (only
`tests/integration/test_e2e_scorer.py` references it). Raw extracted floats flow
straight into `aggregate_scores`/`rank_by_scores`, so a judge emitting `0.8` and one
emitting `8` are averaged without reconciliation, and out-of-range values are never
rejected. The advertised normalization does not run.

**C4. Champion (a dict) is stringified as a Python dict literal in user-facing output.**
`src/certamen/interfaces/render/html_report.py:181,352`
(`str(summary["champion"])`) and `src/certamen/application/workflow/nodes/output.py:193,233`
(`getattr(champion, "display_name", str(champion))`). `tournament_ended.champion`
(`interfaces/cli/main.py:478`) and `gate`/`rank` champions are dicts
(`{name, model_name, provider}` / raw model config), and dicts have no
`display_name` attribute, so the report "Champion" card and the champion line render
`{'name': ..., 'provider': ...}` instead of a clean name.

**C5. Run-data endpoints are unauthenticated even when auth is enabled.**
`src/certamen/interfaces/web/auth/middleware.py:13-17,31-40` — `PROTECTED_PATHS`
lists only `/api/execute`, `/api/validate`, `/ws`. `/api/runs`, `/api/runs/{id}`,
`/api/runs/{id}/events`, and the `/api/runs/{id}/attach` WebSocket fall through
`is_path_protected` to `False`, so the full tournament transcript (prompts, model
responses, costs, champion) is readable without a token when auth is on. Asymmetry
with the protected `/ws` implies a gap, not intent. Severity tempered by the
dev-only / `CERTAMEN_SKIP_AUTH=true` default posture.

**C6. Frontend WebSocket: uncleared reconnect timer + no auth handshake → reconnect storm under auth.**
`frontend/src/hooks/useWebSocket.ts:66-82` — `onclose` schedules
`setTimeout(connect, 3000)` that the effect cleanup never clears (it only calls
`ws.close()`, which itself fires `onclose` → another timer): zombie reconnect on
unmount/url change. Also `onopen` sends only `get_models`/`get_nodes`, never a
`{type:"auth", token}` frame, and the frontend has no token/login flow at all, so
with server auth ON `/ws` is rejected (code 4003) and it reconnects every 3 s
forever. The GUI is functional only under `SKIP_AUTH=true`.

### 🔵 Confirmed

**C7. `--models` / `-m` and `--interactive` / `-i` are dead flags.**
`src/certamen/interfaces/cli/args.py:22-27,51-56` declare them with help text
("Comma-separated list of model keys to run", "Run in interactive mode") but no run
path reads `args["models"]`/`args["interactive"]` — the help promises behavior that
does nothing.

### 🟡/🔵 Latent (real code defect, reachable only via manual GUI wiring — no shipped workflow triggers)

**L1. ImproveNode crashes on string feedback.**
`src/certamen/application/workflow/nodes/generation.py:137-141` does
`model_feedback.items()` assuming a dict, but `PeerReviewNode.evaluations`
(`evaluation.py:120`) emits `dict[str, str]` (values are strings). Wiring
peer-review evaluations → improve `feedback` (exactly what the node tooltip tells
the user to do) makes `model_feedback` a `str` → `AttributeError`, and the call is
inside `parallel_generate.generate_one`, NOT guarded by `safe_generate`'s try/except,
so it aborts the whole `asyncio.gather`. No packaged workflow wires this edge.
(Found independently by 2 scouts.)

**L2. KnowledgeMapNode passes a possibly-raw-dict `judge` to `.generate()`.**
`src/certamen/application/workflow/nodes/synthesis.py:204-216` — unlike
Synthesize/Judge/ExtractInsights/Disagreement nodes, it never calls
`ensure_single_model_instance()` on a connected `judge` model, so a dict champion
(from `gate.champion`/`rank.champion`) → `dict.generate` `AttributeError`. The
builder call is wrapped in `try/except`, so the real effect is a degraded
`[Knowledge map build failed: ...]` rather than a crash. Violates the documented
"any node accepting a model/champion from gate/rank must call
ensure_single_model_instance" rule.

**L3. GateNode conflates empty vs absent feedback.**
`src/certamen/application/workflow/nodes/flow.py:367` — `if feedback is None or
len(feedback) == 0: models = primary`. An empty survivor set (tournament exhausted)
is treated like round-1 (no feedback yet), restarting with the full original model
set. Not reachable in shipped workflows (they use `eliminate.count: 1` and the
`model_count <= 1` short-circuit fires first); reachable via a custom
`count ≥ survivors` config.

**L4. Bare `"529"` error-pattern violates the file's own delimiter rule.**
`src/certamen/shared/constants.py:105` — `"529"` is an undelimited 3-char numeric
substring; the header comment (`constants.py:58-60`) mandates <5-char patterns be
delimiter-wrapped (the same class as the previously-fixed `tps`-in-`https` bug). It
can match `529` inside request ids/token counts and flip a non-retryable error to
retryable `overloaded`. Low reachability: the exception-type map runs first
(`errors.py:166-177`), so the substring path is hit only for typeless string errors.
(Verifier refuted the co-claimed `" 429"`, `"error code: 160"`, `"quota"`, `"retry"`
as rule-compliant.)

### Refuted (checked, not real)

- `_try_extract_fractional_score` (`scoring.py:164-192`): the fractional-division
  branch is unreachable (group(1) is always the non-numeric model name), but the
  `except ValueError` fallback returns the correct `group(2)` score — harmless dead
  sub-branch, no score lost.
- `" 429"`, `"error code: 160"`/`"code: 160"`: delimiter-wrapped / ≥8 chars —
  rule-compliant, not collision-prone.
- Backoff math, cache-key fingerprint, `_ensure_provider_prefix` (no double-prefix),
  `_apply_event_to_run_summary` mapping, `ColoredSmartEdge` memo deps, route paths,
  anonymizer, RankNode, statistics, JSON sanitize, knowledge_store SQL — all checked
  clean.

### Fixes applied (same commit follow-up)

All confirmed + latent findings fixed; `make lint` + `make test` (244 passed) +
frontend `tsc -b`/biome/vitest all green.

- **C1** — `executor._check_termination` now scans all `node_outputs` (mirrors the
  async executor); `execute()` returns `iterations`. New integration test
  `TestEarlyTerminationOnGateDone`. Verified end-to-end: `tournament-elimination.yml`
  now stops at `iterations=6` on the gate signal (was the 20 cap).
- **C2** — `EliminateNode` retains all models (logged) when `scores` is empty,
  instead of eliminating by arbitrary insertion order.
- **C3** — `normalize_score` wired into `_extract_score_for_model` (clamp/normalize
  now runs).
- **C4** — new `shared.mapping_utils.model_display_name`; used in `output.py`
  (markdown/json/text) and `html_report.py` so a dict champion renders its name.
- **C5** — auth middleware switched to deny-by-default (`/api/runs*` etc. now
  protected when auth is enabled).
- **C6** — `useWebSocket` clears the reconnect timer on cleanup and no longer
  reconnects after an intentional close (zombie-loop fix). Frontend has no login
  flow, so the auth-handshake gap remains a documented dev-only limitation.
- **C7** — dead `--models` / `--interactive` flags removed; CLAUDE.md CLI examples
  updated.
- **L1** — `ImproveNode` accepts string feedback (PeerReview evaluations) without
  crashing.
- **L2** — `KnowledgeMapNode` resolves its `judge` via `_resolve_model`
  (`ensure_single_model_instance` on dict input); champion label via
  `model_display_name`.
- **L3** — `GateNode` distinguishes absent feedback (round 1 → primary) from empty
  feedback on a later round (exhausted → done) using `current_round`.
- **L4** — `"529"` error pattern delimiter-wrapped to `" 529"` per the file's own
  rule.

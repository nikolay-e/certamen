# Tournament Analysis: CLI + GUI Integration for Certamen

## Problem

Need to support running certamen as both CLI and GUI:

- CLI runs tournament → produces results → GUI displays them beautifully
- OR GUI launches tournament directly
- GUI must show ALL data: every prompt, every model response, every intermediate stage, every score, plus diagrams

## Constraints

- Existing aiohttp web server at `src/certamen/interfaces/web/server.py` (visual workflow editor)
- Existing CLI at `src/certamen/interfaces/cli/main.py` (tournament + workflow + gui commands)
- Tournament emits ~72 LLM calls for 4 models (PHASE 1 → INTERROGATION → DIVERGENCE → ELIMINATION × N → SYNTHESIS)
- Reports already saved to disk (`outputs/`, JSON + Markdown)
- Knowledge map persisted to SQLite

## Round 1 — Frontend/UX

**Structure: single SPA, three tabs, two modes.** Tabs: `Tournament`, `Workflow Editor` (existing xyflow), `Knowledge Bank` (SQLite browser). Header toggle: **Live** (WebSocket stream, "now playing" banner) vs **Replay** (loaded from `outputs/*.json`, scrubber timeline like Langfuse trace trees — [Langfuse data model](https://langfuse.com/docs/observability/data-model)). Same component tree, different data source — avoids two UIs.

**Layout: 3-pane master-detail.** Top: phase progress strip (Initial → Interrogation → Divergence → Elimination R1..N → Synthesis), each segment shows models-remaining + cumulative cost. Left rail: **diamond bracket** as custom SVG via xyflow (already a dep) — reject `@g-loot/react-tournament-brackets` since elimination is score-based, not 1v1 ([g-loot brackets](https://github.com/g-loot/react-tournament-brackets)). Center: node detail with prompt/response in tabs, NOT side-by-side by default (responses 2-5K tokens; side-by-side breaks at <1600px). Right: score heatmap (models × rubric, color = z-score). Following [LangSmith Preview](https://docs.langchain.com/langsmith/evaluation-concepts): full prompt/response on click, not always.

**Divergence→Convergence:** animate as a collapsing Sankey — N parallel streams widen in divergence, narrow at each elimination boundary. Use **visx** (D3 primitives, React-native, Sankey + Heatmap built-in, ~40KB) over raw d3 (fights React) and recharts (no Sankey/heatmap). Reject Plotly (300KB+).

**Diff view on demand only.** "Compare" button on any two responses opens a modal with `react-diff-viewer` split mode ([praneshr/react-diff-viewer](https://github.com/praneshr/react-diff-viewer); inspired by [LLM Comparator](https://github.com/PAIR-code/llm-comparator)). Not in main flow.

**Hidden by default:** raw JSON, system prompts (collapsible per-phase), token usage, retries, judge chain-of-thought. Cost lives in footer ribbon (always: $X.XX / Yk tokens), full table in modal. The "show everything" trap: 72 calls × 5K tokens = 360K tokens of text — dumping that kills render perf and cognition. Default = phase summaries; drill down for raw.

## Round 1 — Live Streaming

**Code reality check.** `WebSocketEventBridge.publish()` exists and `tournament.py` stores `self.event_handler`, but **no domain code ever calls it** (`grep self.event_handler. domain/` returns empty; `_InternalEventHandler.publish` is a no-op pass). The pipe is built; nothing flows through it. Step 1 is instrumenting `tournament.py` to actually emit events at phase boundaries and per-LLM-call.

**Transport: keep WebSocket, do not add SSE.** WS is already wired (`server.py:223`, auth, rate-limit, CSP `connect-src ws:`, broadcast fan-out under `_clients_lock`). SSE would be theoretically cleaner for this unidirectional firehose ([SSE beats WS for 95% of real-time apps](https://dev.to/polliog/server-sent-events-beat-websockets-for-95-of-real-time-apps-heres-why-a4l), [LogRocket comparison](https://blog.logrocket.com/server-sent-events-vs-websockets/)) — but rebuilding auth+origin checks for a second transport is pure cost. Tradeoff accepted: WS pays a minor complexity tax for bidirectional control (pause/cancel tournament from GUI, which CLI-only flow can't do).

**CLI → GUI signaling.** Current CLI never touches the GUI server. Cleanest path: CLI writes events to `outputs/<run_id>/events.jsonl` (append-only), and the GUI server `tail -f`s that file via `aionotify`/polling, re-broadcasting over WS. This decouples CLI lifetime from server lifetime — no PID files, no IPC sockets, survives CLI crash. PID file in `outputs/<run_id>/run.pid` only for "is it still running?" health check. SQLite polling is wrong here (write amplification, lock contention with knowledge_map.db).

**Attach to in-progress run.** Yes, via the JSONL-as-log pattern: GUI requests `/api/runs/<id>/attach`, server streams existing lines then continues live tail. This mirrors LangSmith's "pending runs" model where traces stream as a tree of runs to a collector that downstream clients subscribe to ([LangSmith observability](https://www.langchain.com/langsmith/observability), [tracing deep-dive](https://medium.com/@aviadr1/langsmith-tracing-deep-dive-beyond-the-docs-75016c91f747)).

**Reconnect/replay.** Each event gets monotonic `seq`. WS client sends `{"type":"resume","run_id":X,"last_seq":N}` on reconnect; server replays JSONL from `seq>N` then resumes live. Same primitive solves "open GUI 5min late".

**Is real-time necessary?** For a 30-min run, "reload after finish" is genuinely defensible — reports are already on disk. But the stated goal ("show ALL data: every prompt, every intermediate stage") implies users want to *watch reasoning unfold*, not autopsy it. Real-time is justified; just don't over-engineer the transport.

Sources: [SSE vs WebSocket 2026](https://www.nimbleway.com/blog/server-sent-events-vs-websockets-what-is-the-difference-2026-guide), [Ably WS vs SSE](https://ably.com/blog/websockets-vs-sse), [LangSmith architecture](https://medium.com/@aviadr1/langsmith-tracing-deep-dive-beyond-the-docs-75016c91f747).

## Round 1 — Storage Architect

### Current State (audit findings)

- In-memory: `ModelComparison` accumulates `previous_answers` (list[dict]), `evaluation_history`, `feedback_history`, `criticism_history`, `eliminated_models`, `evaluation_scores`, `cost_tracker`, `interrogation_context`, `_disagreement_reports`, `_knowledge_map` — never written until tournament end.
- On disk: `provenance.save_to_file` dumps three files per run (`certamen_<ts>_champion_solution.md`, `_provenance.json`, `_complete_history.json`) plus `_knowledge_map.md`. SQLite (`certamen_knowledge.db`) holds **only** cross-run knowledge claims/disagreements/branches — not the run itself.
- Live signal: `EventHandler.publish(event_name, data)` exists but the CLI uses `_InternalEventHandler` (no-op). `WebSocketEventBridge` already converts events to JSON frames.

### Recommendation: **JSONL event stream + SQLite index, NOT either alone**

Crash-safety, append-only auditability, and trivial live-tail are exactly the properties Langfuse/Phoenix provide via Postgres+ClickHouse — overkill for a local desktop tool. The hybrid pattern (events as ground truth, SQLite as derived index) is what Langfuse v3 effectively does at scale ([ClickHouse](https://clickhouse.com/blog/langfuse-and-clickhouse-a-new-data-stack-for-modern-llm-applications)) and what local-first projects converge on ([Terse Systems](https://tersesystems.com/blog/2020/11/26/queryable-logging-with-blacklite/)).

Critique of obvious choices:

- **SQLite-only**: writers block readers under WAL during a 10-min run with 72 LLM calls — fine, but you lose `tail -f` debuggability and have to bolt on triggers for live GUI. Reinventing pub/sub on a relational DB.
- **JSONL-only**: GUI must scan whole file to render "tournament #42 round 3 scores" — O(n) per query. Dead on arrival once you have 50+ runs.
- **Postgres/ClickHouse** (Langfuse path): absurd for a CLI tool the user runs locally. The whole reason Langfuse needed it was multi-tenant cloud scale.

### Schema sketch

**Layout** (per run, under `outputs/runs/<run_id>/`):

```
events.jsonl          # append-only, one event per line
manifest.json         # run metadata (question, models, config snapshot, status)
champion.md           # final markdown (existing)
provenance.json       # final JSON (existing)
```

**Plus** workspace-wide `outputs/index.sqlite` (separate from `certamen_knowledge.db`):

```sql
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, started_at INT, ended_at INT,
  question TEXT, status TEXT, champion_model TEXT,
  total_cost REAL, num_models INT, events_path TEXT
);
CREATE TABLE events (
  run_id TEXT, seq INT, ts REAL, phase TEXT, model TEXT,
  event_type TEXT, payload_offset INT, payload_len INT,
  PRIMARY KEY(run_id, seq)
);  -- payload_offset/len point into events.jsonl, no duplication
CREATE INDEX idx_events_phase ON events(run_id, phase);
```

**Event shape** (extends existing `EventHandler.publish` contract):

```json
{"seq": 47, "ts": 1745625600.123, "run_id": "20260425_143022_a1b2",
 "phase": "ELIMINATION_R2", "event_type": "llm_response",
 "model": "claude-sonnet", "anon": "LLM2",
 "payload": {"prompt": "...", "response": "...", "tokens_in": 1240,
             "tokens_out": 890, "cost_usd": 0.018, "latency_ms": 4231,
             "parent_seq": 46}}
```

Event types map to existing emission points: `tournament_started`, `phase_started`, `prompt_built`, `llm_request`, `llm_response`, `score_extracted`, `feedback_received`, `interrogation_qa`, `disagreement_detected`, `model_eliminated`, `knowledge_map_built`, `synthesis_complete`, `tournament_ended`.

### Read/Write paths

**CLI write**: replace `_InternalEventHandler` no-op with `JsonlEventHandler` that (a) `fsync`-appends to `events.jsonl`, (b) inserts a row into `events` table with byte offset, (c) calls existing `provenance.save_to_file` at run end. Knowledge-map SQLite stays as-is (orthogonal concern).

**GUI read** — three modes, picked by query:

1. **Live tail** (active run): GUI opens websocket; CLI's `EventHandler` fans out to both JSONL writer and the existing `WebSocketEventBridge`. No polling. The `seq` field gives gap detection on reconnect.
2. **Historical browse** (run list, search): query `index.sqlite` — millisecond response, no JSONL parsing.
3. **Run replay** (deep dive into one run): `mmap` the JSONL or `seek(offset)` per event row — random access without loading whole file.

Reject websocket-for-everything: keeping live and historical paths separate avoids a stateful server holding open file handles for closed runs ([Phoenix uses OTLP push + UI pull](https://arize.com/docs/phoenix/tracing/llm-traces); [OpenInference span attribute conventions](https://arize-ai.github.io/openinference/spec/) are a useful naming reference for `phase`/`model`/`event_type` to avoid bikeshedding).

### Critical caveats

- **Don't** put events in `certamen_knowledge.db`. That table is cross-run knowledge; mixing concerns breaks the "delete a run, knowledge survives" property.
- **Don't** store full prompt/response text in SQLite rows — bloats the index, defeats the point. Offsets only.
- **Do** version the event schema (`schema_version` field on every event) — you will change it, JSONL is the only format where schema migration is `jq` instead of `ALTER TABLE`.

Sources:

- [Langfuse + ClickHouse architecture](https://clickhouse.com/blog/langfuse-and-clickhouse-a-new-data-stack-for-modern-llm-applications)
- [Langfuse self-hosting infrastructure](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse)
- [Blacklite: SQLite as queryable log store](https://tersesystems.com/blog/2020/11/26/queryable-logging-with-blacklite/)
- [Event-sourced JSONL + SQLite read cache RFC](https://github.com/paperclipai/paperclip/issues/801)
- [JSONL→SQLite ingestion pipeline pattern](https://github.com/openclaw/openclaw/issues/7783)
- [OpenInference semantic conventions for LLM spans](https://arize-ai.github.io/openinference/spec/)
- [Phoenix tracing overview](https://arize.com/docs/phoenix/tracing/llm-traces)

## Round 1 — Process Orchestration

**Single process or two?** Recommend **two processes sharing a filesystem/SQLite store**, *not* GUI-subprocessing the CLI. Tournament runs are long (~72 calls, minutes, $1+) and stateful; spawning `subprocess.Popen("certamen --config ...")` from aiohttp loses asyncio context, breaks structured logging, and forces stdout-scraping for progress. Better: CLI writes to `outputs/<run_id>/` and SQLite (`certamen_knowledge.db` already exists); GUI is a *read-mostly viewer* of that directory. For "launch from GUI" (mode b), reuse the existing `AsyncExecutor` pattern at `server.py:60` — call `Certamen.run_tournament()` in-process inside an `asyncio.create_task`, broadcast events via the WebSocket bridge already there. This matches MLflow's split: tracking server is a viewer, training is wherever you run it ([MLflow](https://mlflow.org/docs/latest/tracking/)).

**Run ID scheme.** Current code uses `datetime.now().strftime("%Y%m%d_%H%M%S")` (`history.py:108`, `tournament.py:946`). Two CLI invocations in the same second collide (already visible — dual log files per second in `cli_output/`). **Switch to `YYYYMMDD_HHMMSS_<8-hex>`** (timestamp prefix preserves chronological `ls` sort, suffix breaks ties). Optional `--name` flag stores a human label as metadata, never as the directory name.

**Past runs storage.** Promote `outputs/<run_id>/` to canonical: `manifest.json` (question, models, cost, status, started_at, finished_at), `report.md`, `transcript.jsonl` (every prompt+response), `knowledge_map.md`. GUI lists runs via `GET /api/runs` (scan dir + manifest), detail via `GET /api/runs/{id}`. No new DB needed — filesystem-as-database, like MLflow's `mlruns/` default.

**Auth.** Yes — JWT middleware already exists (`auth/middleware.py`). Tournament launch endpoint MUST require auth + a per-user daily $ cap enforced server-side (read `total_cost` from `comparison`). Anonymous read-only viewing of *completed* runs is acceptable; launching is admin-only.

**Worth the complexity?** For mode (c) — CLI now, GUI viewer later — **yes, cheap**: just a directory scanner + Markdown renderer. For mode (b) launch-from-GUI: **defer**. Celery/Redis is overkill ([Toptal](https://www.toptal.com/python/orchestrating-celery-python-background-jobs)); in-process `asyncio.Task` + WebSocket progress ([ichaoran](https://www.ichaoran.com/posts/2024-11-26-long-running-task-app/)) is sufficient at single-tenant scale.

Sources: [MLflow Tracking](https://mlflow.org/docs/latest/tracking/), [Long-running task patterns](https://www.ichaoran.com/posts/2024-11-26-long-running-task-app/), [Celery background jobs](https://www.toptal.com/python/orchestrating-celery-python-background-jobs).

## Round 2 — Adversary

The consensus has reinvented Langfuse for one user. Strip it.

**Simplest viable architecture: append-only JSONL → static HTML report.** Tournament writes `events.jsonl` (already proposed). When run completes, `certamen render <run_id>` produces `report.html` — a single self-contained file with embedded JSON, collapsible `<details>` per phase/call, client-side filter via 50 lines of vanilla JS. No server, no WS, no SQLite index, no auth, no JWT, no $ caps, no async task manager. `python -m http.server` if you want it browsable. MLflow's `mlruns/` does exactly this; static-site generators ship zero JS by default and beat SPAs on load time and complexity ([Hygraph SSG vs SPA](https://hygraph.com/blog/difference-spa-ssg-ssr), [Academind](https://academind.com/articles/dynamic-vs-static-vs-spa)).

**Consensus is premature optimization.** 70% of products fail by scaling before validating ([Stackify](https://stackify.com/premature-optimization-evil/), [eAmped 2026](https://www.eamped.com/how-to-scale-startup-guide-2026/)). Round 1 specifies WebSocket reconnect-with-seq, mmap'd JSONL random access, OpenInference span conventions, visx Sankey animations — for *zero current users*. Langfuse's own docs say start small, focus on vitals ([Langfuse FAQ](https://langfuse.com/faq/all/llm-observability)). Existing tools (`toolong`, `json-tail`) already render JSONL in browser/terminal with zero code ([Textualize/toolong](https://github.com/textualize/toolong), [json-tail](https://github.com/egeozcan/json-tail)) — wire one of those instead of building a new SPA.

**Real-time is a fiction.** A 30-min run doesn't need WS. Tail the JSONL with `tail -f outputs/<run>/events.jsonl | jq` — that's the journalctl experience asked for, free. If user wants progress, print `phase: ELIMINATION_R2 [3/4 models]` to stdout. Done.

**Concrete deletion: cut "launch tournament from GUI" entirely.** It drags in JWT auth, $ caps, in-process asyncio task lifecycle, cancel/pause semantics, WS bidirectionality, CSRF — for a single-user dev tool where `certamen --config` already works. The proponents admit "defer" but still itemize the infra. Delete the endpoint, delete the auth requirement on launch, delete `_clients_lock` fan-out. GUI = read-only static viewer of `outputs/`. Ship in a weekend instead of a quarter.

Sources: [Hygraph SPA vs SSG](https://hygraph.com/blog/difference-spa-ssg-ssr), [Stackify premature optimization](https://stackify.com/premature-optimization-evil/), [eAmped scaling 2026](https://www.eamped.com/how-to-scale-startup-guide-2026/), [Langfuse minimal observability](https://langfuse.com/faq/all/llm-observability), [toolong JSONL viewer](https://github.com/textualize/toolong), [json-tail browser tail](https://github.com/egeozcan/json-tail), [Academind static vs SPA](https://academind.com/articles/dynamic-vs-static-vs-spa).

## Round 2 — Proponent

**Unified plan.** JSONL-as-truth + per-workspace SQLite index, two cooperating processes, single SPA with Live/Replay toggle. This is the same shape Phoenix ships locally (SQLite default in `~/.phoenix/`, [Phoenix persistence](https://docs.arize.com/phoenix/deployment/persistence)) and the same split MLflow uses (filesystem `mlruns/` + viewer server, [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)). Langfuse only moved to S3+ClickHouse+Redis at multi-tenant scale ([Langfuse v3 evolution](https://langfuse.com/blog/2024-12-langfuse-v3-infrastructure-evolution)) — irrelevant for one-user desktop.

**Layout (frozen):**

```
outputs/<run_id>/{events.jsonl, manifest.json, run.pid, champion.md, provenance.json}
outputs/index.sqlite        # run list + (run_id, seq) → byte_offset
certamen_knowledge.db       # untouched, cross-run
```

Run ID: `YYYYMMDD_HHMMSS_<8hex>` via `secrets.token_hex(4)`.

**Event schema (frozen v1):** `{schema_version:1, seq, ts, run_id, phase, event_type, model, anon, parent_seq, payload}`. 13 event types from Round 1 Storage section.

**3 critical implementation steps, in order:**

1. **Wire emission** in `src/certamen/core/tournament.py` — `self.event_handler.publish(...)` at every `_run_phase`, `_call_llm`, `_score`, `_eliminate`, `_synthesize` boundary. Without this, every other piece is dead. Replace `_InternalEventHandler` no-op with `JsonlEventHandler(run_dir)` that fsync-appends + inserts SQLite index row.
2. **Tail bridge** in `interfaces/web/server.py`: new `RunTailer` using `aiofiles.open(events.jsonl)` + `os.stat` size polling at 200ms (cross-platform; macOS FSEvents fight `aiofiles`, [aiohttp+aiofiles](https://hackersandslackers.com/async-requests-aiohttp-aiofiles/)). Re-broadcast through existing `WebSocketEventBridge`. Add `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/attach?from_seq=N`.
3. **Frontend Tournament tab** with Live/Replay toggle. MVP: phase strip + bracket SVG + center detail pane only. Sankey/heatmap deferred.

**MVP slice (1-2 weeks):** steps 1+2 above + minimal React tab rendering phase progress and click-to-expand prompt/response. Ships day 7. No Sankey, no diff modal, no auth changes.

**Refuting the Adversary:**

- *"Static HTML is enough"*: rejects the stated goal — "watch reasoning unfold" requires sub-second updates during a 30-min run. `tail -f | jq` is not a UI a human reads while 4 LLMs argue.
- *"Don't reinvent toolong"*: toolong is a TUI; the user wants a graphical bracket + Sankey + heatmap. Different artifact.
- *"Cut launch-from-GUI"*: agreed — Round 1 already deferred mode (b). MVP keeps it deleted; the JWT/$ cap infra stays dormant.
- *"WebSocket is overkill"*: WS is *already wired* (`server.py:223`). Building the static-HTML alternative is *more* code than `tail -f → ws.send_str()`. The cheapest path is the existing pipe.
- *"Premature"*: step 1 (wire `publish`) is required regardless of UI choice — even the Adversary's static report needs the events to exist.

## Final Synthesis

### TL;DR

Ship in 4 phases over ~3 weeks. The *only* universal precondition is **emitting events from `tournament.py`** — the bridge is wired, nothing is being broadcast. Everything else is UI debate. Adversary wins the YAGNI argument on `mode (b)` (launch from GUI) and on `index.sqlite`; Proponent wins on real-time WS (the pipe already exists, abandoning it is *more* work).

### Recommendation: Diamond approach (mirrors the tournament itself)

**Phase 0 — Event emission (3-5 days, non-negotiable):**

- Replace `_InternalEventHandler.publish` no-op with `JsonlEventHandler` that fsync-appends to `outputs/<run_id>/events.jsonl`.
- Run ID = `YYYYMMDD_HHMMSS_<8hex>` (fixes existing collisions in `cli_output/`).
- Insert `self.event_handler.publish(...)` calls at the 13 emission points (phase boundaries, every `_call_llm`, score extraction, elimination, knowledge map, synthesis).
- Event schema v1 frozen. Include `schema_version` field.
- This step is required by *every* downstream UI choice. Skip the SQLite index for now — Adversary is right that it's premature for <50 runs.

**Phase 1 — Static HTML report (2 days):**

- New CLI command: `certamen render <run_id>` → single self-contained `report.html` with embedded JSON, collapsible `<details>` per phase, vanilla JS filter. Auto-generated at end of every tournament run.
- This satisfies "CLI now → review later" with zero server, zero auth. **Ships value to a single user immediately.**

**Phase 2 — Live tail in existing GUI (5-7 days):**

- New tab in existing React SPA: `Tournament`. Reuse the running aiohttp server.
- `RunTailer` polls `events.jsonl` size every 200ms (not aionotify — cross-platform, simpler), re-broadcasts via existing `WebSocketEventBridge`.
- Endpoints: `GET /api/runs`, `GET /api/runs/{id}`, `WS /api/runs/{id}/attach?from_seq=N`.
- Frontend MVP: phase strip + diamond bracket SVG (xyflow, already a dep) + center detail pane (prompt/response in tabs). **No Sankey, no heatmap, no diff modal yet.**
- Mode toggle: Live (WS stream) vs Replay (load from JSONL via REST).

**Phase 3 — Diagrams & launch-from-GUI (deferred, only if validated):**

- Add visx Sankey + heatmap once Phase 2 is in real use.
- Add `POST /api/tournaments` to launch from GUI. Reuse existing JWT auth + add per-user `$` cap. Use in-process `asyncio.create_task`, **never** Celery.

### Key tradeoffs

| Decision | Won | Cost |
|----------|-----|------|
| JSONL only (no SQLite index in MVP) | Adversary | O(n) for run-list when >50 runs — defer index until pain |
| WebSocket (not SSE, not polling-only) | Proponent | WS is *already wired*; abandoning it costs more than using it |
| Static HTML report alongside live GUI | Adversary | Two render paths to maintain — but they share the JSONL data layer |
| Defer launch-from-GUI | Adversary | User asked for it; deferred to Phase 3 with explicit validation gate |
| Defer Sankey/heatmap/diff | Adversary | "Show everything" goal partially met by Phase 2; rest grows on demand |
| 200ms polling instead of inotify | Proponent | Trivial 5-line implementation; cross-platform; Phoenix uses similar |

### Strongest counter-argument (steel-manned)

*"Just do `certamen render` (Phase 1) and stop there. Static HTML auto-opened in browser after a tournament finishes is 90% of the value at 10% of the cost. Live mode is a vanity feature for a tool that runs offline batches."*

**Why we still build Phase 2:** the user explicitly stated the goal is to "*watch reasoning unfold*" across 30-min runs. Static report = post-mortem only. The aiohttp server + xyflow frontend already exist — abandoning them after the recent legacy-package cleanup is its own waste. Phase 2 cost is ~1 week; Phase 1 enables Phase 2 by writing the JSONL events Phase 2 reads.

### What to delete from prior thinking

- **`outputs/index.sqlite`** — defer until run count > 50 (a year+ away for a single user).
- **OpenInference/OTLP semantic conventions** — over-engineering for a closed-world event vocabulary.
- **Sankey/heatmap/diff modal in MVP** — Phase 3 only.
- **Launch-from-GUI in MVP** — Phase 3 only, with auth + $ cap.
- **`schema_version` JSONL migration tooling** — keep the field, ignore the tooling until it bites.

### Sources (final synthesis)

- [Phoenix local persistence (SQLite default)](https://docs.arize.com/phoenix/deployment/persistence)
- [MLflow tracking server architecture](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [Langfuse v3 infra evolution (when to escalate)](https://langfuse.com/blog/2024-12-langfuse-v3-infrastructure-evolution)
- [Stackify on premature optimization](https://stackify.com/premature-optimization-evil/)
- [Hygraph SPA vs SSG tradeoffs](https://hygraph.com/blog/difference-spa-ssg-ssr)

import html
import json
from pathlib import Path
from typing import Any

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Certamen Tournament: {run_id}</title>
<style>
:root {{
  --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e;
  --accent: #58a6ff; --success: #3fb950; --warn: #d29922; --err: #f85149;
  --panel: #161b22; --border: #30363d;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
  background: var(--bg); color: var(--fg); line-height: 1.5;
}}
header {{
  padding: 1rem 2rem; border-bottom: 1px solid var(--border); background: var(--panel);
  position: sticky; top: 0; z-index: 10;
}}
header h1 {{ margin: 0; font-size: 1.2rem; }}
header .meta {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; }}
main {{ padding: 1.5rem 2rem; max-width: 1400px; margin: 0 auto; }}
.summary {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem; margin-bottom: 2rem;
}}
.card {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 0.75rem 1rem;
}}
.card .label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; }}
.card .value {{ font-size: 1.3rem; font-weight: 600; margin-top: 0.25rem; }}
.phase-strip {{
  display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap;
}}
.phase-pill {{
  padding: 0.4rem 0.9rem; background: var(--panel); border: 1px solid var(--border);
  border-radius: 999px; font-size: 0.85rem;
}}
.controls {{ margin: 1rem 0; display: flex; gap: 0.5rem; align-items: center; }}
.controls input, .controls select {{
  background: var(--panel); border: 1px solid var(--border); color: var(--fg);
  padding: 0.4rem 0.6rem; border-radius: 4px; font-family: inherit;
}}
details {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  margin-bottom: 0.5rem; padding: 0.5rem 1rem;
}}
details > summary {{
  cursor: pointer; font-weight: 500; padding: 0.25rem 0; user-select: none;
  display: flex; gap: 0.75rem; align-items: center;
}}
details[open] > summary {{ border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin-bottom: 0.5rem; }}
.tag {{
  font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 3px;
  background: var(--bg); border: 1px solid var(--border);
}}
.tag.phase {{ color: var(--accent); }}
.tag.model {{ color: var(--success); }}
.tag.elim {{ color: var(--err); border-color: var(--err); }}
.tag.synth {{ color: var(--warn); border-color: var(--warn); }}
.seq {{ color: var(--muted); font-size: 0.75rem; min-width: 3rem; }}
pre {{
  background: var(--bg); padding: 0.75rem; border-radius: 4px; overflow-x: auto;
  font-size: 0.8rem; max-height: 600px; white-space: pre-wrap; word-break: break-word;
}}
.kv {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; font-size: 0.85rem; }}
.kv dt {{ color: var(--muted); }}
.kv dd {{ margin: 0; }}
.hidden {{ display: none !important; }}
.cost {{ color: var(--warn); }}
.section-title {{ margin-top: 2rem; margin-bottom: 0.5rem; color: var(--accent); }}
</style>
</head>
<body>
<header>
<h1>Certamen Tournament</h1>
<div class="meta">
  Run ID: <code>{run_id}</code> · Question: <strong>{question_html}</strong>
</div>
</header>
<main>
<div class="summary">
  <div class="card"><div class="label">Champion</div><div class="value">{champion}</div></div>
  <div class="card"><div class="label">Total Cost</div><div class="value cost">${total_cost:.4f}</div></div>
  <div class="card"><div class="label">LLM Calls</div><div class="value">{llm_count}</div></div>
  <div class="card"><div class="label">Models</div><div class="value">{model_count}</div></div>
  <div class="card"><div class="label">Eliminated</div><div class="value">{eliminated_count}</div></div>
</div>

<div class="phase-strip">{phase_pills}</div>

<h2 class="section-title">Final Answer</h2>
<details open><summary>Synthesized champion answer</summary>
<pre>{final_answer_html}</pre>
</details>

<h2 class="section-title">Event Stream ({event_count} events)</h2>
<div class="controls">
  <label>Filter:
    <select id="filter-type">
      <option value="">All</option>
      <option value="phase_started">Phase boundaries</option>
      <option value="llm_request">LLM requests</option>
      <option value="llm_response">LLM responses</option>
      <option value="model_eliminated">Eliminations</option>
    </select>
  </label>
  <label>Search:
    <input id="filter-text" type="search" placeholder="text in prompts/responses">
  </label>
</div>

<div id="events">
{events_html}
</div>

<h2 class="section-title">Raw Events JSON</h2>
<details><summary>Show full JSONL ({event_count} events)</summary>
<pre id="raw">{raw_jsonl_html}</pre>
</details>
</main>
<script>
const filterType = document.getElementById('filter-type');
const filterText = document.getElementById('filter-text');
const events = document.querySelectorAll('#events > details');
function applyFilters() {{
  const t = filterType.value;
  const q = filterText.value.toLowerCase();
  events.forEach(el => {{
    const matchType = !t || el.dataset.type === t;
    const matchText = !q || (el.dataset.text || '').toLowerCase().includes(q);
    el.classList.toggle('hidden', !(matchType && matchText));
  }});
}}
filterType.addEventListener('change', applyFilters);
filterText.addEventListener('input', applyFilters);
</script>
</body>
</html>
"""


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events = []
    for raw in events_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    question = ""
    champion = "—"
    total_cost = 0.0
    llm_count = 0
    model_count = 0
    eliminated_count = 0
    phases: list[str] = []

    for e in events:
        et = e.get("event_type", "")
        p = e.get("payload", {})
        if et == "tournament_started":
            question = p.get("question", "")
            model_count = len(p.get("models", []))
        elif et == "phase_started":
            phase = p.get("phase", "")
            if phase and phase not in phases:
                phases.append(phase)
        elif et == "llm_response":
            llm_count += 1
            total_cost += float(p.get("cost") or 0.0)
        elif et == "model_eliminated":
            eliminated_count += 1
        elif et == "tournament_ended":
            champion = p.get("champion") or "—"
            if p.get("total_cost") is not None:
                total_cost = float(p["total_cost"])

    return {
        "question": question,
        "champion": champion,
        "total_cost": total_cost,
        "llm_count": llm_count,
        "model_count": model_count,
        "eliminated_count": eliminated_count,
        "phases": phases,
    }


def _render_event(event: dict[str, Any]) -> str:
    seq = event.get("seq", "?")
    et = event.get("event_type", "")
    p = event.get("payload", {})

    title = et
    extra_tags = []
    body_parts = []
    text_for_search = json.dumps(p, default=str).lower()

    if et == "phase_started":
        title = f"PHASE START: {p.get('phase', '?')}"
        extra_tags.append(
            f'<span class="tag phase">{p.get("phase", "")}</span>'
        )
        body_parts.append(_kv(p))
    elif et == "phase_completed":
        title = f"PHASE END: {p.get('phase', '?')}"
        extra_tags.append(
            f'<span class="tag phase">{p.get("phase", "")}</span>'
        )
        body_parts.append(_kv(p))
    elif et == "llm_request":
        title = f"REQUEST → {p.get('anon', p.get('model', '?'))}"
        extra_tags.append(
            f'<span class="tag model">{html.escape(p.get("model", ""))}</span>'
        )
        extra_tags.append(
            f'<span class="tag phase">{html.escape(p.get("phase", ""))}</span>'
        )
        body_parts.append(
            f"<details><summary>Prompt ({p.get('prompt_chars', 0)} chars)</summary>"
            f"<pre>{html.escape(p.get('prompt', ''))}</pre></details>"
        )
    elif et == "llm_response":
        cost = float(p.get("cost") or 0.0)
        is_err = p.get("is_error", False)
        title = f"RESPONSE ← {p.get('anon', p.get('model', '?'))}"
        extra_tags.append(
            f'<span class="tag model">{html.escape(p.get("model", ""))}</span>'
        )
        extra_tags.append(
            f'<span class="tag phase">{html.escape(p.get("phase", ""))}</span>'
        )
        if cost > 0:
            extra_tags.append(f'<span class="tag cost">${cost:.4f}</span>')
        if is_err:
            extra_tags.append('<span class="tag elim">ERROR</span>')
        content = p.get("content", "") or p.get("error", "")
        body_parts.append(
            f"<details open><summary>Response ({len(content)} chars)</summary>"
            f"<pre>{html.escape(content)}</pre></details>"
        )
        if p.get("tokens_in") or p.get("tokens_out"):
            body_parts.append(
                _kv(
                    {
                        "tokens_in": p.get("tokens_in"),
                        "tokens_out": p.get("tokens_out"),
                        "cost": cost,
                    }
                )
            )
    elif et == "model_eliminated":
        title = f"ELIMINATED: {p.get('anon', '?')}"
        extra_tags.append('<span class="tag elim">ELIM</span>')
        body_parts.append(_kv(p))
    elif et == "tournament_started":
        title = "TOURNAMENT STARTED"
        body_parts.append(_kv(p))
    elif et == "tournament_ended":
        title = "TOURNAMENT ENDED"
        extra_tags.append('<span class="tag synth">FINAL</span>')
        body_parts.append(_kv(p))
    else:
        body_parts.append(_kv(p))

    body = "\n".join(body_parts) if body_parts else ""
    return (
        f'<details data-type="{html.escape(et)}" data-text="{html.escape(text_for_search)}">'
        f"<summary>"
        f'<span class="seq">#{seq}</span>'
        f"<span>{html.escape(title)}</span>"
        f"{''.join(extra_tags)}"
        f"</summary>"
        f"{body}"
        f"</details>"
    )


def _kv(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        if k in ("prompt", "content", "response"):
            continue
        v_str = json.dumps(v, default=str) if not isinstance(v, str) else v
        if len(v_str) > 200:
            v_str = v_str[:200] + "…"
        rows.append(
            f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(v_str)}</dd>"
        )
    return f'<dl class="kv">{"".join(rows)}</dl>'


def _phase_pills(summary: dict[str, Any]) -> str:
    if not summary["phases"]:
        return "<span class='tag muted'>No phase events</span>"
    return "".join(
        f'<span class="phase-pill">{html.escape(p)}</span>'
        for p in summary["phases"]
    )


def _final_answer(run_dir: Path) -> str:
    for name in ("champion_solution.md", "champion.md"):
        p = run_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "(final answer not found in run dir)"


def render_run_to_html(run_dir: Path, run_id: str | None = None) -> str:
    run_dir = Path(run_dir)
    if run_id is None:
        run_id = run_dir.name
    events_path = run_dir / "events.jsonl"
    events = _read_events(events_path)
    summary = _summarize(events)

    raw_jsonl = (
        events_path.read_text(encoding="utf-8") if events_path.exists() else ""
    )
    events_html = "\n".join(_render_event(e) for e in events)

    return _HTML_TEMPLATE.format(
        run_id=html.escape(run_id),
        question_html=html.escape(summary["question"] or "(unknown)"),
        champion=html.escape(str(summary["champion"])),
        total_cost=summary["total_cost"],
        llm_count=summary["llm_count"],
        model_count=summary["model_count"],
        eliminated_count=summary["eliminated_count"],
        phase_pills=_phase_pills(summary),
        final_answer_html=html.escape(_final_answer(run_dir)),
        event_count=len(events),
        events_html=events_html,
        raw_jsonl_html=html.escape(raw_jsonl),
    )


def write_run_report(run_dir: Path, output_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    if output_path is None:
        output_path = run_dir / "report.html"
    html_text = render_run_to_html(run_dir)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path

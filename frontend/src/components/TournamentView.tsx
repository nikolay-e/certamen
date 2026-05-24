import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface RunSummary {
  run_id: string;
  status: string;
  question: string | null;
  champion: string | null;
  total_cost: number;
  model_count: number;
  event_count: number;
  started_at: number | null;
  ended_at: number | null;
}

interface CertamenEvent {
  schema_version: number;
  seq: number;
  ts: number;
  run_id: string;
  event_type: string;
  payload: Record<string, unknown>;
}

const PHASE_ORDER = [
  "INITIAL",
  "INTERROGATION",
  "DIVERGENCE",
  "ELIMINATION_R1",
  "ELIMINATION_R2",
  "ELIMINATION_R3",
  "ELIMINATION_R4",
  "SYNTHESIS",
];

function formatTime(ts: number | null): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function formatCost(cost: number): string {
  return `$${cost.toFixed(4)}`;
}

function formatQuestion(question: string): string {
  return question.length > 80 ? `${question.slice(0, 80)}…` : question;
}

function StatusBadge({ status }: Readonly<{ status: string }>) {
  const colors: Record<string, string> = {
    completed: "var(--success-green)",
    running: "var(--accent-orange)",
    empty: "var(--text-tertiary)",
    missing: "var(--error-red)",
    unknown: "var(--text-tertiary)",
  };
  return (
    <span
      style={{
        background: colors[status] || colors.unknown,
        color: "#000",
        padding: "2px 8px",
        borderRadius: 3,
        fontSize: "0.7rem",
        textTransform: "uppercase",
        fontWeight: 600,
      }}
    >
      {status}
    </span>
  );
}

function PhaseStrip({ events }: Readonly<{ events: CertamenEvent[] }>) {
  const seenPhases = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) {
      if (e.event_type === "phase_started" || e.event_type === "phase_completed") {
        const phase = e.payload.phase as string | undefined;
        if (phase) set.add(phase);
      }
    }
    return set;
  }, [events]);

  const completedPhases = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) {
      if (e.event_type === "phase_completed") {
        const phase = e.payload.phase as string | undefined;
        if (phase) set.add(phase);
      }
    }
    return set;
  }, [events]);

  const allPhases = useMemo(() => {
    const out = [...PHASE_ORDER.filter((p) => seenPhases.has(p))];
    for (const p of seenPhases) {
      if (!out.includes(p)) out.push(p);
    }
    return out;
  }, [seenPhases]);

  if (allPhases.length === 0) {
    return <div className="phase-strip-empty">Waiting for tournament to start…</div>;
  }

  return (
    <div className="phase-strip">
      {allPhases.map((phase) => {
        const done = completedPhases.has(phase);
        const active = !done && seenPhases.has(phase);
        let phaseClass = "";
        if (done) phaseClass = "phase-done";
        else if (active) phaseClass = "phase-active";
        return (
          <div key={phase} className={`phase-pill ${phaseClass}`}>
            {phase}
          </div>
        );
      })}
    </div>
  );
}

function EventBody({ event }: Readonly<{ event: CertamenEvent }>) {
  const p = event.payload;
  if (event.event_type === "llm_request" && p.prompt) {
    const promptText = p.prompt as string;
    return <pre className="event-body">{promptText}</pre>;
  }
  if (event.event_type === "llm_response" && (p.content || p.error)) {
    const bodyText = (p.content ?? p.error) as string;
    return <pre className="event-body">{bodyText}</pre>;
  }
  return <pre className="event-body">{JSON.stringify(p, null, 2)}</pre>;
}

function EventCard({ event }: Readonly<{ event: CertamenEvent }>) {
  const [open, setOpen] = useState(false);
  const p = event.payload;

  const titleMap: Record<string, string> = {
    tournament_started: "TOURNAMENT STARTED",
    tournament_ended: "TOURNAMENT ENDED",
    phase_started: `PHASE START: ${(p.phase as string | undefined) ?? "?"}`,
    phase_completed: `PHASE END: ${(p.phase as string | undefined) ?? "?"}`,
    llm_request: `REQUEST → ${(p.anon as string | undefined) ?? (p.model as string | undefined) ?? "?"}`,
    llm_response: `RESPONSE ← ${(p.anon as string | undefined) ?? (p.model as string | undefined) ?? "?"}`,
    model_eliminated: `ELIMINATED: ${(p.anon as string | undefined) ?? "?"}`,
  };

  const title = titleMap[event.event_type] || event.event_type;
  const isError = event.event_type === "llm_response" && p.is_error === true;
  const cost = (p.cost as number) || 0;
  const phaseTag = p.phase as string | undefined;
  const modelTag = p.model as string | undefined;

  return (
    <details
      className={`event-card event-${event.event_type}`}
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary>
        <span className="event-seq">#{event.seq}</span>
        <span className="event-title">{title}</span>
        {phaseTag && <span className="tag tag-phase">{phaseTag}</span>}
        {modelTag && <span className="tag tag-model">{modelTag}</span>}
        {cost > 0 && <span className="tag tag-cost">{formatCost(cost)}</span>}
        {isError && <span className="tag tag-error">ERROR</span>}
        <span className="event-time">{formatTime(event.ts)}</span>
      </summary>
      {open && <EventBody event={event} />}
    </details>
  );
}

function RunList({
  runs,
  selectedId,
  onSelect,
  onRefresh,
}: Readonly<{
  runs: RunSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}>) {
  return (
    <div className="run-list">
      <div className="run-list-header">
        <h3>Tournament Runs</h3>
        <button onClick={onRefresh} type="button">
          Refresh
        </button>
      </div>
      {runs.length === 0 ? (
        <div className="run-list-empty">
          No runs yet. Run <code>certamen --config config.yml</code>.
        </div>
      ) : (
        <ul>
          {runs.map((run) => (
            <li key={run.run_id} className={selectedId === run.run_id ? "selected" : ""}>
              <button type="button" className="run-item-btn" onClick={() => onSelect(run.run_id)}>
                <div className="run-list-row">
                  <code className="run-id">{run.run_id}</code>
                  <StatusBadge status={run.status} />
                </div>
                <div className="run-list-meta">
                  {run.question ? formatQuestion(run.question) : "(no question)"}
                </div>
                <div className="run-list-stats">
                  {run.event_count} events · {formatCost(run.total_cost)} ·{" "}
                  {run.champion ? `Winner: ${run.champion}` : "No champion"}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function TournamentView() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [events, setEvents] = useState<CertamenEvent[]>([]);
  const [filter, setFilter] = useState<string>("");
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [liveStatus, setLiveStatus] = useState<"idle" | "connecting" | "live" | "ended" | "error">(
    "idle",
  );
  const wsRef = useRef<WebSocket | null>(null);

  const fetchRuns = useCallback(async () => {
    try {
      const r = await fetch("/api/runs");
      const data = await r.json();
      setRuns(data.runs || []);
    } catch (e) {
      console.error("Failed to fetch runs:", e);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  // Connect to selected run
  useEffect(() => {
    if (!selectedId) {
      setEvents([]);
      return;
    }
    setEvents([]);
    setLiveStatus("connecting");

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const wsProtocol = globalThis.location.protocol === "https:" ? "wss:" : "ws:";
    const safeRunId = encodeURIComponent(selectedId);
    const wsUrl = `${wsProtocol}//${globalThis.location.host}/api/runs/${safeRunId}/attach?from_seq=0`;
    const ws = new WebSocket(wsUrl); // NOSONAR - URL is derived from window.location (same origin), not user input
    wsRef.current = ws;

    ws.onopen = () => setLiveStatus("live");
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "event" && msg.event) {
          setEvents((prev) => [...prev, msg.event as CertamenEvent]);
        } else if (msg.type === "ended") {
          setLiveStatus("ended");
        } else if (msg.type === "error") {
          setLiveStatus("error");
        }
      } catch (err) {
        console.error("Bad message", err);
      }
    };
    ws.onerror = () => setLiveStatus("error");
    ws.onclose = () => {
      if (liveStatus === "live") setLiveStatus("ended");
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, liveStatus]);

  const filteredEvents = useMemo(() => {
    return events.filter((e) => {
      if (typeFilter && e.event_type !== typeFilter) return false;
      if (filter) {
        const text = JSON.stringify(e.payload).toLowerCase();
        if (!text.includes(filter.toLowerCase())) return false;
      }
      return true;
    });
  }, [events, filter, typeFilter]);

  const summary = useMemo(() => {
    let totalCost = 0;
    let llmCalls = 0;
    let eliminated = 0;
    let champion: string | null = null;
    let question: string | null = null;
    for (const e of events) {
      if (e.event_type === "llm_response") {
        llmCalls++;
        totalCost += (e.payload.cost as number) || 0;
      } else if (e.event_type === "model_eliminated") {
        eliminated++;
      } else if (e.event_type === "tournament_ended") {
        champion = (e.payload.champion as string) || null;
        if (e.payload.total_cost != null) {
          totalCost = e.payload.total_cost as number;
        }
      } else if (e.event_type === "tournament_started") {
        question = (e.payload.question as string) || null;
      }
    }
    return { totalCost, llmCalls, eliminated, champion, question };
  }, [events]);

  return (
    <div className="tournament-view">
      <RunList runs={runs} selectedId={selectedId} onSelect={setSelectedId} onRefresh={fetchRuns} />
      <div className="tournament-main">
        {selectedId ? (
          <>
            <div className="tournament-header">
              <div>
                <h2>{selectedId}</h2>
                <div className="tournament-question">{summary.question || "(loading…)"}</div>
              </div>
              <div className="tournament-status">
                <span className={`live-indicator live-${liveStatus}`}>
                  ● {liveStatus.toUpperCase()}
                </span>
              </div>
            </div>

            <div className="tournament-summary">
              <div className="card">
                <div className="card-label">Champion</div>
                <div className="card-value">{summary.champion || "—"}</div>
              </div>
              <div className="card">
                <div className="card-label">Total Cost</div>
                <div className="card-value">{formatCost(summary.totalCost)}</div>
              </div>
              <div className="card">
                <div className="card-label">LLM Calls</div>
                <div className="card-value">{summary.llmCalls}</div>
              </div>
              <div className="card">
                <div className="card-label">Eliminated</div>
                <div className="card-value">{summary.eliminated}</div>
              </div>
              <div className="card">
                <div className="card-label">Events</div>
                <div className="card-value">{events.length}</div>
              </div>
            </div>

            <PhaseStrip events={events} />

            <div className="event-controls">
              <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                <option value="">All event types</option>
                <option value="phase_started">Phase boundaries</option>
                <option value="llm_request">LLM requests</option>
                <option value="llm_response">LLM responses</option>
                <option value="model_eliminated">Eliminations</option>
              </select>
              <input
                type="search"
                placeholder="Search events…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              />
              <span className="event-count">
                {filteredEvents.length} / {events.length} events
              </span>
            </div>

            <div className="event-stream">
              {filteredEvents.map((e) => (
                <EventCard key={e.seq} event={e} />
              ))}
            </div>
          </>
        ) : (
          <div className="tournament-empty">Select a run from the left to view its details.</div>
        )}
      </div>
    </div>
  );
}

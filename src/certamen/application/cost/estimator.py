from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import litellm

from certamen.domain.errors import ConfigurationError
from certamen.shared.validation.context import estimate_token_count

LOCAL_ZERO_COST_PROVIDERS = {"ollama"}

# Per-call output-token priors (min / expected / max-fraction-of-cap) by stage.
# "expected" is anchored on measured healthy flagship runs (~2.3k completion tok
# for generate/improve; reasoning tokens are already folded into completion_tokens
# by the providers, so they are baked into these figures). These are ASSUMPTIONS,
# surfaced verbatim in the report so the number is never falsely precise.
OUTPUT_PRIORS: dict[str, tuple[int, int]] = {
    # stage: (min_tokens, expected_tokens)   max = per-model max_tokens cap
    "generate": (600, 2300),
    "interrogate_questions": (150, 450),
    "interrogate_answers": (400, 1300),
    "diverge_improve": (600, 2300),
    "peer_review": (400, 1300),
    "converge_improve": (600, 2300),
}

# Representative forwarded-response size used to size the INPUT of later stages
# (improve/peer_review concatenate prior responses). Tied to generate output.
RESP_MIN, RESP_EXP = 800, 2300

# Small fixed per-call template + system-prompt overhead (tokens). Measured from
# the interrogation/evaluation/improvement templates (all <= ~150 tok).
TEMPLATE_OVERHEAD = 150


@dataclass
class PriceRow:
    input_per_token: float
    output_per_token: float
    source: str


@dataclass
class CompetitorEstimate:
    key: str
    model_name: str
    provider: str
    price: PriceRow
    input_tokens: int
    cost_min: float
    cost_expected: float
    cost_max: float


@dataclass
class CostEstimate:
    workflow_name: str
    n_competitors: int
    divergence_iterations: int
    peer_review_calls: int
    converge_calls: int
    total_calls: int
    stalled_total_calls: int
    competitors: list[CompetitorEstimate]
    total_min: float
    total_expected: float
    total_max: float
    stalled_expected: float
    assumptions: list[str] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.competitors)


def _price_per_1m(price_per_token: float) -> float:
    return price_per_token * 1_000_000


def resolve_price(
    model_name: str,
    provider: str,
    overrides: dict[str, Any],
) -> PriceRow:
    if provider.lower() in LOCAL_ZERO_COST_PROVIDERS:
        return PriceRow(0.0, 0.0, source=f"{provider} (local, $0)")

    override = overrides.get(model_name) or overrides.get(
        model_name.split("/")[-1]
    )
    if override is not None:
        in_1m = float(override["input_per_1m"])
        out_1m = float(override["output_per_1m"])
        return PriceRow(
            in_1m / 1_000_000,
            out_1m / 1_000_000,
            source="config price_overrides",
        )

    for candidate in _price_key_candidates(model_name, provider):
        entry = litellm.model_cost.get(candidate)
        if entry and entry.get("input_cost_per_token") is not None:
            return PriceRow(
                float(entry["input_cost_per_token"]),
                float(entry.get("output_cost_per_token", 0.0)),
                source=f"litellm.model_cost['{candidate}']",
            )

    raise ConfigurationError(
        f"No price found for model '{model_name}' (provider '{provider}'). "
        f"It is absent from litellm.model_cost and from config 'price_overrides'. "
        f"Add it under price_overrides, e.g.:\n"
        f"  price_overrides:\n"
        f"    {model_name}:\n"
        f"      input_per_1m: 2.0\n"
        f"      output_per_1m: 6.0\n"
        f"Refusing to silently price it at $0."
    )


def _price_key_candidates(model_name: str, provider: str) -> list[str]:
    bare = model_name.split("/")[-1]
    return [
        model_name,
        bare,
        f"{provider}/{bare}",
    ]


def _find_nodes(
    workflow: dict[str, Any], node_type: str
) -> list[dict[str, Any]]:
    return [n for n in workflow["nodes"] if n.get("type") == node_type]


def _node_prop(node: dict[str, Any], key: str, default: Any) -> Any:
    return node.get("properties", {}).get(key, default)


def _extract_question(workflow: dict[str, Any]) -> str:
    for node in workflow["nodes"]:
        if node.get("id") == "question" and node.get("type") == "simple/text":
            texts = node.get("properties", {}).get("texts", [])
            if texts:
                return str(texts[0])
    return ""


def _simulate_rounds(
    n: int, max_rounds: int, eliminate_count: int, scoring_works: bool
) -> tuple[int, int, int]:
    """Return (divergence_iterations, peer_review_calls, converge_calls)."""
    it = 0
    count = n
    div_iters = 0
    pr_calls = 0
    ci_calls = 0
    while True:
        it += 1
        div_iters += (
            1  # divergence re-runs every iteration reached (no memoization)
        )
        if count <= 1:
            break
        if it >= max_rounds:
            break
        pr_calls += count
        survivors = max(count - eliminate_count, 1) if scoring_works else count
        ci_calls += survivors
        if scoring_works:
            count = survivors
    return div_iters, pr_calls, ci_calls


def estimate_cost(
    workflow: dict[str, Any],
    price_overrides: dict[str, Any] | None = None,
) -> CostEstimate:
    overrides = price_overrides or {}
    question = _extract_question(workflow)

    llm_nodes = _find_nodes(workflow, "simple/llm")
    n = len(llm_nodes)
    if n == 0:
        raise ConfigurationError(
            "Cost estimate: workflow has no 'simple/llm' competitor nodes."
        )

    gate_nodes = _find_nodes(workflow, "flow/gate")
    max_rounds = (
        int(_node_prop(gate_nodes[0], "max_rounds", 10)) if gate_nodes else 1
    )
    elim_nodes = _find_nodes(workflow, "tournament/eliminate")
    eliminate_count = (
        int(_node_prop(elim_nodes[0], "count", 1)) if elim_nodes else 1
    )
    has_interrogation = bool(_find_nodes(workflow, "tournament/interrogate"))
    has_peer_review = bool(_find_nodes(workflow, "tournament/peer_review"))
    n_diverge_improve = len(_find_nodes(workflow, "tournament/improve"))

    div_iters, pr_calls, ci_calls = _simulate_rounds(
        n, max_rounds, eliminate_count, scoring_works=True
    )
    stalled_div, stalled_pr, stalled_ci = _simulate_rounds(
        n, max_rounds, eliminate_count, scoring_works=False
    )

    # Per-model call counts (attribution). Survivor-round calls (peer_review,
    # converge) have unknown identity → attributed evenly across competitors.
    pairs_examiner_per_iter = n - 1  # each model examines n-1 others per iter
    per_model = {
        "generate": div_iters,
        "interrogate_questions": div_iters * pairs_examiner_per_iter
        if has_interrogation
        else 0,
        "interrogate_answers": div_iters * pairs_examiner_per_iter
        if has_interrogation
        else 0,
        # diamond has both a divergence-improve and a convergence-improve node;
        # divergence-improve runs once per model per iteration.
        "diverge_improve": div_iters if n_diverge_improve >= 1 else 0,
        "peer_review": (pr_calls / n) if has_peer_review else 0,
        "converge_improve": (ci_calls / n) if n_diverge_improve >= 2 else 0,
    }

    total_calls = round(sum(per_model.values()) * n)
    stalled_total = round(
        (
            stalled_div
            + (
                2 * stalled_div * pairs_examiner_per_iter
                if has_interrogation
                else 0
            )
            + (stalled_div if n_diverge_improve >= 1 else 0)
            + (stalled_pr / n if has_peer_review else 0)
            + (stalled_ci / n if n_diverge_improve >= 2 else 0)
        )
        * n
    )

    competitors: list[CompetitorEstimate] = []
    for node in llm_nodes:
        props = node.get("properties", {})
        model_name = str(props.get("model_name", ""))
        provider = str(props.get("provider", ""))
        key = str(props.get("name") or node.get("id") or model_name)
        max_tokens = int(props.get("max_tokens", 4096) or 4096)

        price = resolve_price(model_name, provider, overrides)
        q_tokens = (
            estimate_token_count(question, model_name) if question else 0
        )

        comp = _estimate_competitor(
            key=key,
            model_name=model_name,
            provider=provider,
            price=price,
            q_tokens=q_tokens,
            n=n,
            max_tokens=max_tokens,
            per_model_calls=per_model,
        )
        competitors.append(comp)

    total_min = sum(c.cost_min for c in competitors)
    total_expected = sum(c.cost_expected for c in competitors)
    total_max = sum(c.cost_max for c in competitors)
    stalled_scale = (stalled_total / total_calls) if total_calls else 1.0
    stalled_expected = total_expected * stalled_scale

    assumptions = [
        f"{n} competitors; workflow re-runs the divergence phase every iteration "
        f"(no memoization) → {div_iters} divergence iterations.",
        f"Call count: {total_calls} (healthy) … up to {stalled_total} "
        f"(if peer-review scores fail to parse → no elimination → runs to "
        f"max_rounds={max_rounds}).",
        "Input tokens are counted exactly (litellm.token_counter) for round-1 "
        "question text; forwarded-response sizes in later stages are MODELED "
        f"(~{RESP_EXP} tok expected).",
        "Output tokens are MODELED per stage (reasoning tokens are billed as "
        "output and are already folded into the expected figures). max = each "
        "call saturates its model's max_tokens cap.",
        "Prompt caching assumed OFF (matches runtime: certamen sends no "
        "cache_control), so the question is billed at full input on every resend.",
        "Prices from litellm.model_cost unless overridden; ollama priced at $0.",
    ]

    return CostEstimate(
        workflow_name=str(workflow.get("name", "workflow")),
        n_competitors=n,
        divergence_iterations=div_iters,
        peer_review_calls=pr_calls,
        converge_calls=ci_calls,
        total_calls=total_calls,
        stalled_total_calls=stalled_total,
        competitors=competitors,
        total_min=total_min,
        total_expected=total_expected,
        total_max=total_max,
        stalled_expected=stalled_expected,
        assumptions=assumptions,
    )


def _stage_input_tokens(stage: str, q_tokens: int, n: int, resp: int) -> int:
    others = max(n - 1, 0)
    table = {
        "generate": q_tokens,
        "interrogate_questions": q_tokens + 2 * resp,
        "interrogate_answers": q_tokens + resp,
        "diverge_improve": q_tokens + others * resp,
        "peer_review": q_tokens + n * resp,
        "converge_improve": q_tokens + others * resp,
    }
    return table[stage] + TEMPLATE_OVERHEAD


def _estimate_competitor(
    key: str,
    model_name: str,
    provider: str,
    price: PriceRow,
    q_tokens: int,
    n: int,
    max_tokens: int,
    per_model_calls: dict[str, float],
) -> CompetitorEstimate:
    def run(resp: int, out_mode: str) -> tuple[float, int]:
        cost = 0.0
        in_tokens = 0
        for stage, calls in per_model_calls.items():
            if calls <= 0:
                continue
            in_tok = _stage_input_tokens(stage, q_tokens, n, resp)
            out_min, out_exp = OUTPUT_PRIORS[stage]
            # A call can never emit more than its max_tokens cap, so priors are
            # clamped to it — otherwise a tight cap makes "max" fall below
            # "expected".
            if out_mode == "min":
                out_tok = min(out_min, max_tokens)
            elif out_mode == "max":
                out_tok = max_tokens
            else:
                out_tok = min(out_exp, max_tokens)
            cost += calls * (
                in_tok * price.input_per_token
                + out_tok * price.output_per_token
            )
            in_tokens += round(calls * in_tok)
        return cost, in_tokens

    cost_min, _ = run(RESP_MIN, "min")
    cost_expected, in_tokens = run(RESP_EXP, "expected")
    cost_max, _ = run(RESP_EXP, "max")

    return CompetitorEstimate(
        key=key,
        model_name=model_name,
        provider=provider,
        price=price,
        input_tokens=in_tokens,
        cost_min=cost_min,
        cost_expected=cost_expected,
        cost_max=cost_max,
    )


def format_estimate(estimate: CostEstimate) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"=== Cost estimate: {estimate.workflow_name} ===")
    lines.append("")
    header = f"{'Competitor':<22}{'in $/1M':>9}{'out $/1M':>10}{'min':>9}{'expected':>11}{'max':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for c in estimate.competitors:
        lines.append(
            f"{c.key[:21]:<22}"
            f"{_price_per_1m(c.price.input_per_token):>9.2f}"
            f"{_price_per_1m(c.price.output_per_token):>10.2f}"
            f"{c.cost_min:>9.2f}"
            f"{c.cost_expected:>11.2f}"
            f"{c.cost_max:>9.2f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<22}{'':>9}{'':>10}"
        f"{estimate.total_min:>9.2f}"
        f"{estimate.total_expected:>11.2f}"
        f"{estimate.total_max:>9.2f}"
    )
    lines.append("")
    lines.append(
        f"LLM calls: {estimate.total_calls} (healthy) … "
        f"{estimate.stalled_total_calls} (scoring-stall worst case)"
    )
    lines.append(
        f"Worst-case cost if scoring stalls: ~${estimate.stalled_expected:.2f}"
    )
    lines.append(
        f"Total input tokens (modeled): ~{estimate.total_input_tokens:,}"
    )
    lines.append("")
    lines.append("Assumptions (this is a bounded estimate, not a quote):")
    for a in estimate.assumptions:
        lines.append(f"  - {a}")
    lines.append("")
    return "\n".join(lines)

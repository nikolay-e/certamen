from typing import Any

from certamen_core.domain.model_selection import select_model_by_capacity


def select_model_with_highest_context(models: dict[str, Any]) -> str | None:
    return select_model_by_capacity(models, include_max_tokens=False)


# Models are discovered dynamically from LiteLLM and Ollama API.
# For Ollama: queries local server at OLLAMA_BASE_URL/api/tags
# For cloud providers: uses litellm.models_by_provider as source of truth
# This empty dict is kept for backward compatibility with config loading.
MODELS: dict[str, dict[str, Any]] = {}

# Default retry settings
RETRY: dict[str, Any] = {
    "max_attempts": 3,
    "initial_delay": 10,
    "max_delay": 60,
}

# Default feature flags
FEATURES: dict[str, Any] = {
    "save_reports_to_disk": True,
    "deterministic_mode": True,
    "judge_model": None,  # None = use peer evaluation, or specify model key for single judge
    "knowledge_bank_model": "leader",  # "leader" = use current leader, or specify model key
    "llm_compression": False,  # Disabled - user pays for full context
    "compression_model": None,  # None = auto-select model with highest context window
    "synthesis_enabled": True,  # Enable synthesis phase after champion determination
    # Knowledge extraction features
    "interrogation_enabled": True,  # Adversarial cross-examination between models
    "interrogation_max_questions": 4,  # Questions per model pair
    "disagreement_investigation_enabled": True,  # Deep-dive into model disagreements
    "confidence_calibration_enabled": True,  # Append confidence tags to initial prompts
    "knowledge_map_enabled": True,  # Build structured knowledge map after tournament
    "deep_extraction_enabled": False,  # Recursive sub-tournaments on exploration branches (expensive)
    "deep_extraction_depth": 2,  # Max recursion depth for deep extraction
    "persistence_enabled": True,  # Persist knowledge maps to SQLite across tournaments
    "persistence_db_path": "certamen_knowledge.db",  # Path to knowledge store database
    "interrogation_rounds": 1,  # Number of interrogation rounds (round 2 probes round 1 findings)
}

# Default prompts (JSON-like structured format)
PROMPTS: dict[str, Any] = {
    "confidence_calibrated": {
        "content": (
            "For each significant claim in your response, append a confidence tag:\n"
            "[HIGH] — You have strong evidence or this is well-established\n"
            "[MEDIUM] — Reasonable inference but could be wrong\n"
            "[LOW] — Educated guess, limited evidence\n"
            "[UNCERTAIN] — You genuinely don't know but are offering a possibility\n\n"
            "Also identify at the end of your response:\n"
            "KNOWN_UNKNOWNS: Things relevant to this question that you know you don't know\n"
            "ASSUMPTIONS: Premises you're assuming but haven't verified\n"
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "confidence",
        },
    },
    "initial": {
        "content": (
            "Before answering, briefly identify the fundamental principles underlying this problem. "
            "Consider at least 2-3 genuinely different approaches before committing to one. "
            "Analyze the problem from multiple perspectives using evidence-based reasoning and common sense. "
            "Be aware of your inherent biases and actively work to counteract them. "
            "Don't be constrained by current trends or prevailing narratives - trends change, "
            "human priorities shift, and emotional reactions should not limit objective analysis. "
            "Identify the core and outline several distinct, well-reasoned strategies to efficiently SOLVE THE PROBLEM, "
            "grounded in scientific principles and common sense. Avoid unfalsifiable claims, "
            "pseudoscience, and speculative nonsense. Think critically and don't hesitate to "
            "challenge assumptions as the question can be self-restrictive. "
            "When proposing tactics or metrics with incomplete evidence, DO NOT drop them; label them explicitly as heuristics. "
            "Use an evidence strength tag for every non-obvious claim: [STRONG]/[MODERATE]/[WEAK]/[ANECDOTAL], "
            "and include a confidence estimate (e.g., 0.2-0.8). "
            "If a precise number lacks a source, convert it to an operational range or test rather than deleting it."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "initial_response",
        },
    },
    "feedback": {
        "content": (
            "Provide feedback that will allow the one to improve. "
            "Identify the most insightful, evidence-based ideas that stand out. "
            "Be aware of confirmation bias and other cognitive biases when evaluating. "
            "Don't favor responses that merely align with popular sentiment - value independent "
            "thinking that challenges current trends when evidence supports it. "
            "Distinguish between verifiable insights and mere speculation. "
            "Note which elements rely on common consensus versus proven methodology. "
            "Explicitly list high-utility details (e.g., concrete metrics, micro-behaviors) that are weakly evidenced but useful; "
            "recommend preserving them as labeled heuristics rather than removing them. "
            "Penalize deletion of unique, actionable specifics if they can be retained with uncertainty labels."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "feedback",
        },
    },
    "improvement": {
        "content": (
            "Improve the answer using feedback, grounding it in scientific evidence and "
            "practical reasoning. Recognize and mitigate your own biases in the analysis. "
            "Don't let emotional appeals or fashionable opinions constrain rigorous thinking. "
            "CRITICAL: Resist the pressure to converge toward other responses. Your independent "
            "perspective is your most valuable contribution. If your analysis contradicts others, "
            "investigate WHY — do NOT simply yield to the majority. The goal is maximum knowledge "
            "extraction, not consensus. "
            "Before revising, identify what unique insight in YOUR original answer is absent from others' "
            "responses. Preserve it — it may be the most valuable contribution. "
            "Make the key verifiable insights the central thesis. "
            "Rebuild the entire argument around this main point, removing all generic claims, "
            "unsubstantiated speculation, and secondary details. "
            "Do NOT discard high-utility specifics; instead, reframe them with evidence tags and confidence levels. "
            "Translate uncited precise numbers into operational ranges or decision rules. "
            "Append a 'Heuristics Annex' that retains weakly evidenced but useful tactics, clearly labeled. "
            "Produce a short Change Log explaining what was generalized/relocated and why. "
            "After writing your improved answer, reconsider: What did you simplify? What perspective is missing?"
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "improvement",
        },
    },
    "evaluate": {
        "content": (
            "You are an editor judging analytical depth, scientific rigor and common sense. "
            "Evaluate the REASONING PROCESS, not just final conclusions. "
            "A response with novel, well-evidenced insights should score higher even if its structure is less polished. "
            "How insightful and evidence-based is this answer? Does it rely on proven methodology and sound "
            "reasoning, or does it resort to speculation and unfalsifiable claims? "
            "Be aware that all analysis contains inherent biases - evaluate whether the answer "
            "acknowledges and addresses its own potential biases. Does it demonstrate intellectual "
            "independence by challenging prevailing trends when warranted, or does it merely "
            "echo popular sentiment? Be critical of both originality and factual grounding. "
            "Reward explicit evidence tags, confidence reporting, and retention of useful heuristics in an annex. "
            "Down-rank answers that delete actionable details without justification or fail to provide a Change Log. "
            "Check that precise claims without sources were converted to ranges/tests rather than removed. "
            "For EACH response, before scoring, identify: "
            "(a) STRONGEST CONTRIBUTION — what unique insight or evidence does this response offer that others miss? "
            "(b) CRITICAL WEAKNESS — what is the most significant gap, error, or unfounded claim? "
            "(c) SPECIFIC IMPROVEMENT — one concrete, actionable suggestion to strengthen this response."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "evaluation",
        },
    },
    "synthesis": {
        "content": (
            "You are synthesizing multiple expert analyses into the most comprehensive answer possible. "
            "Your task is to create a unified answer that preserves EVERY unique finding, piece of evidence, "
            "perspective, and nuance from ALL responses below. Nothing valuable should be lost. "
            "Before synthesizing, analyze the responses through three lenses: "
            "(1) CONSENSUS — where do all or most experts converge? These points form your foundation. "
            "(2) CONTESTED POINTS — where do experts disagree? For each disagreement, investigate WHY they "
            "diverge (different assumptions? different evidence? different reasoning methods?) and present "
            "both sides with evidence quality assessment. Disagreement often signals the frontier of knowledge. "
            "Do NOT average competing views — preserve the tension and let the reader decide. "
            "(3) UNIQUE FINDINGS — what appears in only ONE response? These minority insights may be the most "
            "valuable, representing knowledge that only one model's training captured. Preserve them explicitly. "
            "Combine strengths of all responses while eliminating redundancy. "
            "The final synthesis should be richer and more complete than any individual response."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "synthesis",
        },
    },
}

REASONING_PERSPECTIVES: list[str] = [
    (
        "Analyze using first-principles reasoning. Break the problem down to its fundamental components. "
        "Challenge every assumption. What must be true for any solution to work?"
    ),
    (
        "Analyze through historical precedents and analogies. What similar situations have occurred? "
        "What lessons from adjacent domains apply here? What patterns repeat?"
    ),
    (
        "Take a contrarian stance. Challenge the conventional wisdom on this topic. "
        "What is everyone getting wrong? What risks and failure modes are being ignored?"
    ),
    (
        "Use systems thinking. Focus on second-order effects, feedback loops, and unintended consequences. "
        "How do the parts interact? What emerges from the whole that isn't visible in the parts?"
    ),
    (
        "Apply decision-theoretic analysis. What are the key trade-offs? "
        "What information would change the optimal decision? What are the irreversible choices?"
    ),
    (
        "Reason from the perspective of implementation and practice. What are the real-world obstacles? "
        "What would actually work vs. what sounds good in theory? Focus on actionable specifics."
    ),
]

# Default Knowledge Bank settings
KNOWLEDGE_BANK: dict[str, Any] = {
    "enabled": True,
    "similarity_threshold": 0.75,  # Cosine similarity threshold for duplicate detection
    "max_insights": 100,  # Maximum insights to keep (LRU eviction)
}


def get_defaults() -> dict[str, Any]:
    return {
        "models": MODELS,
        "retry": RETRY,
        "features": FEATURES,
        "prompts": PROMPTS,
        "reasoning_perspectives": REASONING_PERSPECTIVES,
        "knowledge_bank": KNOWLEDGE_BANK,
    }

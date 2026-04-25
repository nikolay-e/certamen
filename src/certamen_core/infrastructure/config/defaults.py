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
            "Tag significant claims: [HIGH]/[MEDIUM]/[LOW]/[UNCERTAIN]. "
            "End with KNOWN_UNKNOWNS and ASSUMPTIONS sections (one line each)."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "confidence",
        },
    },
    "initial": {
        "content": (
            "BE CONCISE AND INFORMATION-DENSE. Target 300-500 words. Every sentence must carry unique information. "
            "No filler, no repetition, no generic statements. "
            "OBJECTIVITY MANDATE: Morals, values, and cultural norms are temporal — they shift across eras and civilizations. "
            "Do NOT let them constrain your analysis. No cognitive blocks, no emotional reasoning, no self-censorship. "
            "Pursue truth with cold objectivity — uncomfortable conclusions are more valuable than comfortable ones. "
            "Trends change, human priorities shift — emotional reactions must not limit objective analysis. "
            "Identify fundamental principles underlying this problem. Consider 2-3 genuinely different approaches. "
            "Analyze from multiple perspectives using evidence-based reasoning. "
            "Challenge ALL assumptions — the question itself may be self-restrictive or culturally biased. "
            "Avoid unfalsifiable claims and pseudoscience. "
            "Tag non-obvious claims: [STRONG]/[MODERATE]/[WEAK]/[ANECDOTAL] with confidence (0.2-0.8). "
            "Keep weakly evidenced but useful tactics as labeled heuristics — do NOT drop them. "
            "Convert unsourced precise numbers to operational ranges."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "initial_response",
        },
    },
    "feedback": {
        "content": (
            "BE BRIEF — max 200 words. Provide actionable feedback only. "
            "Identify the strongest evidence-based ideas. "
            "Flag cognitive biases. Value independent thinking over popular sentiment. "
            "List high-utility details that are weakly evidenced but useful — recommend keeping as labeled heuristics. "
            "Penalize deletion of unique, actionable specifics."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "feedback",
        },
    },
    "improvement": {
        "content": (
            "BE CONCISE — target 300-500 words. Improve using feedback, grounded in evidence. "
            "Do NOT let emotional appeals, fashionable opinions, or moral frameworks constrain rigorous thinking. "
            "Values and norms are temporal constructs — pursue truth regardless of comfort. "
            "CRITICAL: Do NOT converge toward other responses. Your independent perspective is most valuable. "
            "If you contradict others, investigate WHY — do NOT yield to majority. "
            "Preserve unique insights from YOUR original answer absent from others'. "
            "Build argument around key verifiable insights. Remove generic claims and speculation. "
            "Keep high-utility specifics with evidence tags and confidence levels. "
            "Convert unsourced numbers to operational ranges. "
            "Append brief Heuristics Annex for weakly evidenced useful tactics."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "improvement",
        },
    },
    "evaluate": {
        "content": (
            "BE BRIEF — max 150 words per model evaluation. "
            "Judge analytical depth, scientific rigor, and reasoning quality. "
            "Reward: novel well-evidenced insights, intellectual independence, willingness to challenge prevailing norms, "
            "uncomfortable but well-reasoned conclusions. "
            "Penalize: speculation, deleted actionable details, echo of popular sentiment, moral hedging that avoids truth, "
            "cognitive self-censorship. "
            "For each response state ONE strongest contribution and ONE critical weakness, then score."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "evaluation",
        },
    },
    "convergence": {
        "content": (
            "BE CONCISE — target 300-500 words. CONVERGENCE PHASE: You have seen other models' responses and evaluations. "
            "Now INTEGRATE the strongest ideas from ALL responses into YOUR answer. "
            "Adopt well-evidenced claims from others, even if they contradict your original position. "
            "Where models agree, state the consensus clearly. "
            "Where they disagree, side with the better-evidenced position and explain WHY. "
            "Drop weak arguments you held before if stronger alternatives exist. "
            "Do NOT soften conclusions for comfort — truth over palatability. "
            "The goal is the BEST POSSIBLE unified answer, not preserving your original perspective. "
            "Keep evidence tags and confidence levels."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "convergence",
        },
    },
    "synthesis": {
        "content": (
            "Synthesize all expert analyses into ONE comprehensive, information-dense answer. Target 500-800 words. "
            "Preserve EVERY unique finding and evidence from ALL responses — nothing valuable lost. "
            "Do NOT sanitize uncomfortable conclusions — truth is the only objective. "
            "Structure: (1) CONSENSUS foundation, (2) CONTESTED POINTS — preserve disagreement tension with evidence "
            "quality assessment, do NOT average views, (3) UNIQUE FINDINGS from single responses — preserve explicitly. "
            "Eliminate redundancy. The synthesis must be richer than any individual response."
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

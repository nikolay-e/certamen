"""Default configuration values for Certamen Framework.

All defaults are defined as Python dictionaries to ensure they are properly
included in the package distribution. This avoids path resolution issues
with YAML files when running from repository vs installed package.
"""

from typing import Any


def select_model_with_highest_context(models: dict[str, Any]) -> str | None:
    if not models:
        return None

    best_model_key = None
    best_context = -1

    for model_key, model_obj in models.items():
        # Handle both dict configs and BaseModel instances
        if isinstance(model_obj, dict):
            context_window = model_obj.get("context_window", 0)
        elif hasattr(model_obj, "context_window"):
            context_window = getattr(model_obj, "context_window", 0)
        else:
            context_window = 0

        if context_window and context_window > best_context:
            best_context = context_window
            best_model_key = model_key

    return best_model_key


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
}

# Default prompts (JSON-like structured format)
PROMPTS: dict[str, dict[str, Any]] = {
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
}

# Default Knowledge Bank settings
KNOWLEDGE_BANK: dict[str, Any] = {
    "enabled": True,
    "similarity_threshold": 0.75,  # Cosine similarity threshold for duplicate detection
    "max_insights": 100,  # Maximum insights to keep (LRU eviction)
}

# Default API provider secrets configuration
SECRETS: dict[str, Any] = {
    "providers": {
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "op_path": "op://Shared/OpenAI/api-key",
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY",
            "op_path": "op://Personal/Anthropic/api-key",
        },
        "vertex_ai": {
            "env_var": "VERTEX_AI_API_KEY",
            "op_path": "op://Personal/VertexAI/api-key",
        },
        "xai": {
            "env_var": "XAI_API_KEY",
            "op_path": "op://Personal/XAI/api-key",
        },
        "google": {
            "env_var": "GOOGLE_API_KEY",
            "op_path": "op://Personal/Google/api-key",
        },
        "cohere": {
            "env_var": "COHERE_API_KEY",
            "op_path": "op://Personal/Cohere/api-key",
        },
        "mistral": {
            "env_var": "MISTRAL_API_KEY",
            "op_path": "op://Personal/Mistral/api-key",
        },
    }
}


def get_defaults() -> dict[str, Any]:
    """
    Get all default configuration values.

    Returns:
        Dictionary containing all default configuration sections.
    """
    return {
        "models": MODELS,
        "retry": RETRY,
        "features": FEATURES,
        "prompts": PROMPTS,
        "knowledge_bank": KNOWLEDGE_BANK,
        "secrets": SECRETS,
    }

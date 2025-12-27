from typing import Any


def select_model_by_capacity(
    models: dict[str, Any],
    include_max_tokens: bool = False,
) -> str | None:
    if not models:
        return None

    best_model_key = None
    best_score = -1

    for model_key, model_obj in models.items():
        # Handle both dict configs and BaseModel instances
        if isinstance(model_obj, dict):
            context_window = model_obj.get("context_window", 0) or 0
            max_tokens = model_obj.get("max_tokens", 0) or 0
        else:
            context_window = getattr(model_obj, "context_window", 0) or 0
            max_tokens = getattr(model_obj, "max_tokens", 0) or 0

        score = context_window
        if include_max_tokens:
            score += max_tokens

        if score > best_score:
            best_score = score
            best_model_key = model_key

    return best_model_key

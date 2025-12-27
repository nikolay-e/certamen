import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from arbitrium_core.domain.workflow import BaseNode as DomainBaseNode
from arbitrium_core.domain.workflow import (
    ExecutionContext,
    NodeProperty,
    Port,
    PortType,
)
from arbitrium_core.shared.constants import MAX_MULTI_INPUTS
from arbitrium_core.shared.logging import get_contextual_logger

__all__ = [
    "BaseNode",
    "ExecutionContext",
    "NodeProperty",
    "Port",
    "PortType",
    "build_evaluation_prompt",
    "build_group_outputs",
    "format_responses",
    "merge_indexed_dicts",
    "merge_indexed_lists",
    "parallel_generate",
    "partition_models",
    "rank_by_scores",
    "require_inputs",
    "safe_generate",
    "split_by_rank",
]

logger = get_contextual_logger(__name__)


class BaseNode(DomainBaseNode):
    async def ensure_models_or_empty(
        self, models_input: Any, single: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from arbitrium_core.infrastructure.llm.factory import (
            ensure_model_instances,
            ensure_single_model_instance,
        )

        if single:
            model = await ensure_single_model_instance(models_input, "model")
            if not model:
                return {}, self._get_empty_output()
            return {"model": model}, None
        else:
            models = await ensure_model_instances(models_input)
            if not models:
                return {}, self._get_empty_output()
            return models, None


F = TypeVar("F", bound=Callable[..., Any])


def require_inputs(*required_keys: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(
            self: BaseNode,
            inputs: dict[str, Any],
            context: ExecutionContext,
        ) -> dict[str, Any]:
            valid, error_output = self.validate_required_inputs(
                inputs, *required_keys
            )
            if not valid:
                return error_output
            result: dict[str, Any] = await func(self, inputs, context)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def build_group_outputs(
    groups: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        f"group_{i + 1}": groups[i] if i < len(groups) else {}
        for i in range(MAX_MULTI_INPUTS)
    }


def merge_indexed_dicts(
    inputs: dict[str, Any],
    prefix: str,
    separator: str = "_",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for i in range(1, MAX_MULTI_INPUTS + 1):
        key = f"{prefix}{separator}{i}" if separator else f"{prefix}{i}"
        value = inputs.get(key, {})
        if value:
            result.update(value)
    return result


def merge_indexed_lists(
    inputs: dict[str, Any],
    prefix: str,
    separator: str = "_",
) -> list[Any]:
    result: list[Any] = []
    for i in range(1, MAX_MULTI_INPUTS + 1):
        key = f"{prefix}{separator}{i}" if separator else f"{prefix}{i}"
        value = inputs.get(key, [])
        if value:
            result.extend(value)
    return result


def partition_models(
    models: dict[str, Any],
    scores: dict[str, float],
    check_fn: Callable[[float], bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    passed = {k: v for k, v in models.items() if check_fn(scores.get(k, 0))}
    failed = {
        k: v for k, v in models.items() if not check_fn(scores.get(k, 0))
    }
    return passed, failed


def format_responses(responses: dict[str, str], separator: str = "===") -> str:
    return "\n\n".join(
        f"{separator} {name} {separator}\n{text}"
        for name, text in responses.items()
    )


def build_evaluation_prompt(
    question: str,
    responses: dict[str, str],
    criteria: str = "",
) -> str:
    formatted = format_responses(responses)
    parts = [
        f"Question: {question}",
        "Evaluate each response on a scale of 1-10:",
        formatted,
    ]
    if criteria:
        parts.append(criteria)
    parts.append(
        "Provide a score for each model in the format:\nMODEL_NAME: SCORE/10"
    )
    return "\n\n".join(parts)


async def safe_generate(model: Any, prompt: str) -> tuple[str, bool]:
    try:
        response = await model.generate(prompt)
        if response.is_error():
            logger.warning(
                f"Model generation returned error response: {response}"
            )
            return "", False
        return response.content, True
    except Exception as e:
        logger.error(f"Exception during model generation: {e}", exc_info=True)
        return "", False


async def parallel_generate(
    models: dict[str, Any],
    prompt_fn: Callable[[str, Any], str],
) -> dict[str, str]:
    async def generate_one(key: str, model: Any) -> tuple[str, str]:
        prompt = prompt_fn(key, model)
        content, _ = await safe_generate(model, prompt)
        return key, content

    tasks = [generate_one(k, m) for k, m in models.items()]
    results = await asyncio.gather(*tasks)
    return {k: v for k, v in results if v}


def rank_by_scores(
    models: dict[str, Any],
    scores: dict[str, float],
    reverse: bool = True,
) -> list[str]:
    return sorted(
        models.keys(), key=lambda k: scores.get(k, 0), reverse=reverse
    )


def split_by_rank(
    models: dict[str, Any],
    ranked_keys: list[str],
    keep_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kept = {k: models[k] for k in ranked_keys[:keep_count]}
    removed = {k: models[k] for k in ranked_keys[keep_count:]}
    return kept, removed

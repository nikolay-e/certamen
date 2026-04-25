import json
import re
from typing import Any


def to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        result = obj.model_dump()
        if isinstance(result, dict):
            return result
    if hasattr(obj, "dict"):
        result = obj.dict()
        if isinstance(result, dict):
            return result
    try:
        result = dict(obj)
        if isinstance(result, dict):
            return result
    except (TypeError, ValueError):
        pass
    return {}


def _extract_json_text(text: str) -> str | None:
    # Try to find JSON code block first
    json_block_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1)

    # Try to find bare JSON (starting with { or [)
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    return None


def _is_likely_set_syntax(json_text: str) -> bool:
    return json_text.strip().startswith("{") and ":" not in json_text[:50]


def _try_convert_set_to_list(
    json_text: str,
) -> dict[str, object] | list[object] | None:
    try:
        # Replace first { with [
        list_text = json_text.replace("{", "[", 1)
        # Replace last } with ]
        list_text = "]".join(list_text.rsplit("}", 1))
        result: dict[str, object] | list[object] = json.loads(list_text)
        return result
    except json.JSONDecodeError:
        return None


def extract_json_from_text(
    text: str,
) -> dict[str, object] | list[object] | None:
    if not text or not isinstance(text, str):
        return None

    # Extract JSON text from input
    json_text = _extract_json_text(text)
    if not json_text:
        return None

    # Try to parse as standard JSON
    try:
        result: dict[str, object] | list[object] = json.loads(json_text)
        return result
    except json.JSONDecodeError:
        pass

    # Try to handle set-like syntax (common model error)
    if _is_likely_set_syntax(json_text):
        return _try_convert_set_to_list(json_text)

    return None

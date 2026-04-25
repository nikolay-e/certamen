from typing import Any


def sanitize_for_json(
    obj: Any,
    max_length: int | None = None,
    truncate_suffix: str = "... (truncated)",
) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float)):
        return obj
    elif isinstance(obj, str):
        # Ensure valid UTF-8
        sanitized = obj.encode("utf-8", errors="replace").decode("utf-8")
        if max_length and len(sanitized) > max_length:
            return sanitized[:max_length] + truncate_suffix
        return sanitized
    elif isinstance(obj, dict):
        return {
            sanitize_for_json(
                k, max_length, truncate_suffix
            ): sanitize_for_json(v, max_length, truncate_suffix)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [
            sanitize_for_json(item, max_length, truncate_suffix)
            for item in obj
        ]
    else:
        str_repr = str(obj)
        if max_length and len(str_repr) > max_length:
            return str_repr[:max_length] + truncate_suffix
        return str_repr

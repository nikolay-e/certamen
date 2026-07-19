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
        return dict(obj)
    except (TypeError, ValueError):
        pass
    return {}

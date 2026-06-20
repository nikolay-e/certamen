from typing import Any


def model_display_name(champion: Any) -> str:
    if champion is None:
        return ""
    if isinstance(champion, dict):
        return str(
            champion.get("name")
            or champion.get("display_name")
            or champion.get("model_name")
            or champion
        )
    return str(getattr(champion, "display_name", None) or champion)


def deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

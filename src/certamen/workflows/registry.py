from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

BUILTIN_WORKFLOWS = ("diamond-tournament",)


class BuiltinWorkflowNotFoundError(FileNotFoundError):
    pass


def _builtin_path(name: str) -> Path | None:
    resource = files("certamen.workflows").joinpath(f"{name}.yml")
    with as_file(resource) as path:
        if path.is_file():
            return Path(path)
    return None


def _user_path(name: str) -> Path | None:
    candidate = Path.home() / ".certamen" / "workflows" / f"{name}.yml"
    return candidate if candidate.is_file() else None


def resolve_workflow_path(name_or_path: str) -> Path:
    direct = Path(name_or_path)
    if direct.is_file():
        return direct.resolve()

    builtin = _builtin_path(name_or_path)
    if builtin is not None:
        return builtin

    user = _user_path(name_or_path)
    if user is not None:
        return user

    raise BuiltinWorkflowNotFoundError(
        f"Workflow '{name_or_path}' not found. "
        f"Looked in: file path, built-in registry "
        f"({', '.join(BUILTIN_WORKFLOWS)}), and ~/.certamen/workflows/."
    )


def list_builtin_workflows() -> list[str]:
    return list(BUILTIN_WORKFLOWS)

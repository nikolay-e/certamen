#!/usr/bin/env python3
"""Structure linter for arbitrium-core.

Enforces:
- No .yaml files (use .yml only)
- No SQLite WAL/SHM/journal files tracked in git
- No *_node.py files in domain/workflow/nodes/
- nodes/ compat layer: only __init__.py, registry.py, llm.py (pure re-exports)
- Root facades (executor.py, tournament.py, etc.) must be pure re-exports
- snake_case naming for Python modules under src/
- Symlink policy: README.md → CLAUDE.md allowed, .claude/ allowed
"""

import ast
import re
import subprocess
import sys
from pathlib import Path


def get_git_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(f) for f in result.stdout.strip().split("\n") if f]


def check_yaml_extensions(files: list[Path]) -> list[str]:
    errors = []
    excluded = {".pre-commit-config.yaml", ".secrets.baseline"}
    for f in files:
        if f.suffix == ".yaml" and f.name not in excluded:
            errors.append(f"YAML extension: {f} (use .yml instead)")
    return errors


def check_sqlite_artifacts(files: list[Path]) -> list[str]:
    errors = []
    bad_suffixes = {".db-wal", ".db-shm", ".db-journal"}
    for f in files:
        for suffix in bad_suffixes:
            if str(f).endswith(suffix):
                errors.append(f"SQLite artifact tracked: {f}")
    return errors


def check_node_naming(root: Path) -> list[str]:
    errors = []
    nodes_dir = (
        root / "src" / "arbitrium_core" / "domain" / "workflow" / "nodes"
    )
    if not nodes_dir.exists():
        nodes_dir = (
            root / "src" / "arbitrium" / "domain" / "workflow" / "nodes"
        )
    if not nodes_dir.exists():
        return []

    for f in nodes_dir.glob("*_node.py"):
        errors.append(
            f"Node file with _node suffix: {f} (remove _node suffix)"
        )
    return errors


def is_pure_reexport(filepath: Path) -> bool:
    """Check if a Python file contains only imports and __all__."""
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
    except (SyntaxError, OSError):
        return False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "__all__":
                    continue
            return False
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):
                continue
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            return False

    return True


def check_nodes_compat_layer(root: Path) -> list[str]:
    """Check that nodes/ directory contains only allowed re-export stubs."""
    errors = []
    nodes_dir = root / "src" / "arbitrium_core" / "nodes"
    if not nodes_dir.exists():
        return []

    # Only these files are allowed in the compat layer
    allowed_files = {"__init__.py", "registry.py", "llm.py"}
    for f in nodes_dir.glob("*.py"):
        if f.name not in allowed_files:
            errors.append(
                f"nodes/{f.name} not allowed (only {', '.join(sorted(allowed_files))} permitted)"
            )
        elif not is_pure_reexport(f):
            errors.append(
                f"nodes/{f.name} contains implementation (must be pure re-export)"
            )

    return errors


def check_root_facades(root: Path) -> list[str]:
    """Check that root-level 'duplicate concept' modules are facades only.

    These modules re-export from canonical domain/application locations.
    They must not contain implementation (no class/def) to prevent
    structural ambiguity.
    """
    errors = []
    pkg = root / "src" / "arbitrium_core"
    if not pkg.exists():
        return []

    # These root modules must be pure facades (re-exports only)
    facade_modules = [
        "executor.py",
        "prompts.py",
        "report.py",
        "tournament.py",
        "anonymizer.py",
        "helpers.py",
        "scorer.py",
    ]

    for mod_name in facade_modules:
        mod_path = pkg / mod_name
        if mod_path.exists() and not is_pure_reexport(mod_path):
            errors.append(
                f"Root module {mod_name} must be facade only (no class/def)"
            )

    return errors


def check_snake_case(root: Path) -> list[str]:
    """Check that Python modules under src/ use snake_case."""
    errors = []
    src_dir = root / "src"
    if not src_dir.exists():
        return []

    snake_case_pattern = re.compile(r"^[a-z][a-z0-9_]*\.py$")
    package_pattern = re.compile(r"^[a-z][a-z0-9_]*$")

    excluded_dirs = {".egg-info", "__pycache__", ".mypy_cache"}

    for item in src_dir.rglob("*"):
        if any(excl in str(item) for excl in excluded_dirs):
            continue

        if item.is_file() and item.suffix == ".py":
            if item.name == "__init__.py" or item.name == "__about__.py":
                continue
            if not snake_case_pattern.match(item.name):
                errors.append(
                    f"Non-snake_case module: {item.relative_to(root)}"
                )
        elif item.is_dir():
            if item.name.startswith("__") or item.name.startswith("."):
                continue
            if not package_pattern.match(item.name):
                errors.append(
                    f"Non-snake_case package: {item.relative_to(root)}"
                )

    return errors


def check_symlink_policy(root: Path) -> list[str]:
    """Check symlinks: only README.md → CLAUDE.md and .claude/* are allowed."""
    errors = []

    result = subprocess.run(
        [
            "find",
            str(root),
            "-type",
            "l",
            "-not",
            "-path",
            "*/venv/*",
            "-not",
            "-path",
            "*/.venv/*",
            "-not",
            "-path",
            "*/.git/*",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        symlink = Path(line)
        try:
            rel_path = symlink.relative_to(root)
        except ValueError:
            continue
        rel_str = str(rel_path)

        # Only README.md and .claude/* symlinks are allowed
        if rel_str == "README.md":
            continue
        if rel_str.startswith(".claude/"):
            continue

        errors.append(
            f"Symlink not allowed: {rel_str} (ban policy - use real files)"
        )

    # Verify README.md points to CLAUDE.md
    readme = root / "README.md"
    claude = root / "CLAUDE.md"
    if claude.exists() and readme.exists():
        if not readme.is_symlink():
            errors.append("README.md should be symlink to CLAUDE.md")
        elif readme.resolve() != claude.resolve():
            errors.append(
                f"README.md points to wrong target: {readme.resolve()}"
            )

    return errors


def main() -> int:
    root = Path.cwd()
    files = get_git_tracked_files()

    all_errors: list[str] = []
    all_errors.extend(check_yaml_extensions(files))
    all_errors.extend(check_sqlite_artifacts(files))
    all_errors.extend(check_node_naming(root))
    all_errors.extend(check_nodes_compat_layer(root))
    all_errors.extend(check_root_facades(root))
    all_errors.extend(check_snake_case(root))
    all_errors.extend(check_symlink_policy(root))

    if all_errors:
        print("Structure linting errors:")
        for error in all_errors:
            print(f"  ✗ {error}")
        return 1

    print("✓ Structure linting passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import importlib
import os
import pkgutil
import sys

import certamen

# Minimal valid env so import-time validation in interfaces/web/auth passes
# (mirrors a real deployment); setdefault keeps any real CI-provided values.
os.environ.setdefault("CERTAMEN_JWT_SECRET", "x" * 40)
os.environ.setdefault("CERTAMEN_SKIP_AUTH", "true")
os.environ.setdefault("SKIP_DB_INIT", "true")


def test_every_module_imports() -> None:
    failures: list[str] = []

    def on_error(name: str) -> None:
        failures.append(f"{name}: {sys.exc_info()[1]!r}")

    for module_info in pkgutil.walk_packages(
        certamen.__path__,
        prefix=f"{certamen.__name__}.",
        onerror=on_error,
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            failures.append(f"{module_info.name}: {exc!r}")

    assert not failures, "modules failed to import:\n" + "\n".join(failures)

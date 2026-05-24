from certamen.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)

_NODE_MODULES = (
    "disagreement",
    "evaluation",
    "flow",
    "generation",
    "input",
    "interrogation",
    "knowledge",
    "llm",
    "output",
    "synthesis",
)


def register_all() -> None:
    import importlib

    for module_name in _NODE_MODULES:
        importlib.import_module(
            f"certamen.application.workflow.nodes.{module_name}"
        )


__all__ = ["BaseNode", "ExecutionContext", "Port", "PortType", "register_all"]

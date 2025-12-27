from typing import Any

from arbitrium_core.application.workflow.registry import registry


def list_node_types() -> list[str]:
    return sorted(registry._nodes.keys())


def get_node_schema(node_type: str) -> dict[str, Any] | None:
    node_class = registry._nodes.get(node_type)
    if not node_class:
        return None

    return {
        "type": node_type,
        "display_name": node_class.DISPLAY_NAME,
        "category": node_class.CATEGORY,
        "description": node_class.DESCRIPTION,
        "inputs": [
            {
                "name": port.name,
                "type": port.port_type.value,
                "required": port.required,
                "description": port.description,
            }
            for port in node_class.INPUTS
        ],
        "outputs": [
            {
                "name": port.name,
                "type": port.port_type.value,
                "description": port.description,
            }
            for port in node_class.OUTPUTS
        ],
        "properties": node_class.PROPERTIES,
    }

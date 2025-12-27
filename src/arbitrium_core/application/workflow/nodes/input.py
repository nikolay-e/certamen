from typing import Any

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)
from arbitrium_core.application.workflow.registry import register_node


@register_node
class ModelSelectorNode(BaseNode):
    NODE_TYPE = "tournament/models"
    DISPLAY_NAME = "Models"
    CATEGORY = "Tournament"
    DESCRIPTION = "Collect models for tournament (connect LLM nodes)"

    INPUTS = [
        Port(
            "model_1",
            PortType.MODEL,
            required=False,
            description="Connect LLM node's model_config output here",
        ),
        Port(
            "model_2",
            PortType.MODEL,
            required=False,
            description="Connect another LLM node to compete against model_1",
        ),
    ]
    OUTPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="All connected models bundled together - connect to Generate node",
        )
    ]

    DYNAMIC_INPUTS = {
        "prefix": "model",
        "port_type": "model",
        "min_count": 2,
    }

    PROPERTIES = {
        "selected_models": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
            "options": [],
            "dynamic": True,
            "description": "Alternative: pick from config.yml models instead of connecting LLM nodes",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models: dict[str, Any] = {}

        for key, value in inputs.items():
            if key.startswith("model_") and value and isinstance(value, dict):
                name = value.get("name", key)
                models[name] = value

        if not models:
            selected = self.node_properties.get("selected_models", [])
            models = {k: v for k, v in context.models.items() if k in selected}

        return {"models": models}

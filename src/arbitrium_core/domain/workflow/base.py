from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class PortType(Enum):
    MODELS = "models"
    MODEL = "model"
    RESPONSES = "responses"
    SCORES = "scores"
    RANKINGS = "rankings"
    RESULTS = "results"
    INSIGHTS = "insights"
    BOOLEAN = "boolean"
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    STRING_MATRIX = "string_matrix"
    ANY = "any"


@dataclass
class Port:
    name: str
    port_type: PortType
    required: bool = True
    description: str = ""


@dataclass
class ExecutionContext:
    question: str = ""
    round_num: int = 0
    execution_id: str = ""
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    broadcast: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


class NodeProperty:
    def __init__(self, key: str, type_: type = str, default: Any = None):
        self.key = key
        self.type_ = type_
        self.default = default

    def __get__(self, obj: Any, _objtype: Any = None) -> Any:
        if obj is None:
            return self
        value = obj.node_properties.get(self.key, self.default)
        if value is None:
            return self.default
        try:
            return self.type_(value)
        except (ValueError, TypeError):
            return self.default


class BaseNode(ABC):
    NODE_TYPE: ClassVar[str] = ""
    DISPLAY_NAME: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    HIDDEN: ClassVar[bool] = False
    INPUTS: ClassVar[list[Port]] = []
    OUTPUTS: ClassVar[list[Port]] = []
    PROPERTIES: ClassVar[dict[str, Any]] = {}
    DYNAMIC_INPUTS: ClassVar[dict[str, Any] | None] = None

    def __init__(self, node_id: str, properties: dict[str, Any] | None = None):
        self.node_id = node_id
        self.node_properties = properties or {}
        self._validate_properties()

    def _validate_properties(self) -> None:
        if not self.PROPERTIES:
            return

        schema_keys = set(self.PROPERTIES.keys())
        actual_keys = set(self.node_properties.keys())

        unknown = actual_keys - schema_keys
        if unknown:
            logger.warning(
                f"Node {self.node_id} ({self.NODE_TYPE}): Unknown properties {unknown}. "
                f"Expected properties: {schema_keys}. "
                f"This may indicate a typo or outdated workflow file."
            )

        for key, value in self.node_properties.items():
            if key not in self.PROPERTIES:
                continue

            schema = self.PROPERTIES[key]
            expected_type = schema.get("type")

            if expected_type == "array" and not isinstance(value, list):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be array, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "object" and not isinstance(value, dict):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be object, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "string" and not isinstance(value, str):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be string, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "number" and not isinstance(
                value, (int, float)
            ):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be number, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "boolean" and not isinstance(value, bool):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be boolean, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )

        for key, schema in self.PROPERTIES.items():
            if "default" not in schema and key not in actual_keys:
                logger.warning(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Missing property '{key}' "
                    f"(no default value defined)"
                )

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        pass

    def get_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "node_type": self.NODE_TYPE,
            "display_name": self.DISPLAY_NAME,
            "category": self.CATEGORY,
            "description": self.DESCRIPTION,
            "inputs": [
                {
                    "name": p.name,
                    "port_type": p.port_type.value,
                    "required": p.required,
                    "description": p.description,
                }
                for p in self.INPUTS
            ],
            "outputs": [
                {
                    "name": p.name,
                    "port_type": p.port_type.value,
                    "description": p.description,
                }
                for p in self.OUTPUTS
            ],
            "properties": self.PROPERTIES,
        }
        if self.DYNAMIC_INPUTS:
            schema["dynamic_inputs"] = self.DYNAMIC_INPUTS
        return schema

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        errors = []
        for port in self.INPUTS:
            if port.required and port.name not in inputs:
                errors.append(f"Missing required input: {port.name}")
        return errors

    def validate_required_inputs(
        self, inputs: dict[str, Any], *required_keys: str
    ) -> tuple[bool, dict[str, Any]]:
        missing = []
        for key in required_keys:
            value = inputs.get(key)
            if not value:
                missing.append(key)

        if missing:
            logger.warning(
                f"Node {self.node_id} ({self.NODE_TYPE}) missing required inputs: {missing}"
            )
            return False, self._get_empty_output()

        return True, {}

    def _get_empty_output(self) -> dict[str, Any]:
        mapping_ports = {
            PortType.MODELS,
            PortType.RESPONSES,
            PortType.SCORES,
        }
        return {
            port.name: (
                {}
                if port.port_type in mapping_ports
                else [] if port.port_type == PortType.RANKINGS else ""
            )
            for port in self.OUTPUTS
        }

    async def broadcast_result(
        self, event_type: str, data: dict[str, Any], context: ExecutionContext
    ) -> None:
        if context.broadcast:
            await context.broadcast(
                {
                    "type": event_type,
                    "node_id": self.node_id,
                    "data": data,
                }
            )

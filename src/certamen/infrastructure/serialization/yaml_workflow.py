from pathlib import Path
from typing import Any

import yaml

from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class WorkflowValidationError(Exception):
    pass


def _check_missing_fields(
    data: dict[str, Any], required_fields: list[str]
) -> list[str]:
    return [field for field in required_fields if field not in data]


class WorkflowLoader:
    SUPPORTED_VERSIONS = ["1.0"]
    REQUIRED_FIELDS = ["version", "name", "nodes"]
    NODE_REQUIRED_FIELDS = ["id", "type"]
    EDGE_REQUIRED_FIELDS = ["source", "target"]

    @staticmethod
    def load_from_file(path: str | Path) -> dict[str, Any]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        logger.info("Loading workflow from %s", path)

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise WorkflowValidationError(
                f"Failed to parse YAML file {path}: {e}"
            ) from e

        if not isinstance(data, dict):
            raise WorkflowValidationError(
                f"Invalid YAML structure: expected dict, got {type(data).__name__}"
            )

        WorkflowLoader._validate_workflow(data, path)

        logger.info(
            "Loaded workflow '%s' with %s nodes",
            data["name"],
            len(data["nodes"]),
        )

        return data

    @staticmethod
    def load_from_string(yaml_content: str) -> dict[str, Any]:
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise WorkflowValidationError(f"Failed to parse YAML: {e}") from e

        if not isinstance(data, dict):
            raise WorkflowValidationError(
                f"Invalid YAML structure: expected dict, got {type(data).__name__}"
            )

        WorkflowLoader._validate_workflow(data, "<string>")

        return data

    @staticmethod
    def _validate_workflow(data: dict[str, Any], source: str | Path) -> None:
        missing_fields = _check_missing_fields(
            data, WorkflowLoader.REQUIRED_FIELDS
        )
        if missing_fields:
            raise WorkflowValidationError(
                f"{source}: Missing required fields: {missing_fields}"
            )

        version = data["version"]
        if version not in WorkflowLoader.SUPPORTED_VERSIONS:
            raise WorkflowValidationError(
                f"{source}: Unsupported version {version}. "
                f"Supported: {WorkflowLoader.SUPPORTED_VERSIONS}"
            )

        node_ids = WorkflowLoader._validate_nodes(data["nodes"], source)
        WorkflowLoader._validate_edges(data.get("edges", []), node_ids, source)
        WorkflowLoader._validate_outputs(
            data.get("outputs", []), node_ids, source
        )

    @staticmethod
    def _validate_nodes(nodes: Any, source: str | Path) -> set[str]:
        if not isinstance(nodes, list):
            raise WorkflowValidationError(
                f"{source}: 'nodes' must be a list, got {type(nodes).__name__}"
            )

        if len(nodes) == 0:
            raise WorkflowValidationError(
                f"{source}: Workflow must have at least one node"
            )

        node_ids: set[str] = set()
        for i, node in enumerate(nodes):
            WorkflowLoader._validate_single_node(node, i, node_ids, source)
        return node_ids

    @staticmethod
    def _validate_single_node(
        node: Any, index: int, node_ids: set[str], source: str | Path
    ) -> None:
        if not isinstance(node, dict):
            raise WorkflowValidationError(
                f"{source}: Node {index} must be a dict, got {type(node).__name__}"
            )

        missing_node_fields = _check_missing_fields(
            node, WorkflowLoader.NODE_REQUIRED_FIELDS
        )
        if missing_node_fields:
            raise WorkflowValidationError(
                f"{source}: Node {index} missing required fields: {missing_node_fields}"
            )

        node_id = node["id"]
        if not isinstance(node_id, str) or not node_id.strip():
            raise WorkflowValidationError(
                f"{source}: Node {index} has invalid id: {node_id}"
            )

        if node_id in node_ids:
            raise WorkflowValidationError(
                f"{source}: Duplicate node id: {node_id}"
            )
        node_ids.add(node_id)

        if not isinstance(node["type"], str) or not node["type"].strip():
            raise WorkflowValidationError(
                f"{source}: Node {node_id} has invalid type: {node.get('type')}"
            )

        if "properties" in node and not isinstance(node["properties"], dict):
            raise WorkflowValidationError(
                f"{source}: Node {node_id} properties must be a dict, "
                f"got {type(node['properties']).__name__}"
            )

    @staticmethod
    def _validate_edges(
        edges: Any, node_ids: set[str], source: str | Path
    ) -> None:
        if not isinstance(edges, list):
            raise WorkflowValidationError(
                f"{source}: 'edges' must be a list, got {type(edges).__name__}"
            )

        for i, edge in enumerate(edges):
            WorkflowLoader._validate_single_edge(edge, i, node_ids, source)

    @staticmethod
    def _validate_single_edge(
        edge: Any, index: int, node_ids: set[str], source: str | Path
    ) -> None:
        if not isinstance(edge, dict):
            raise WorkflowValidationError(
                f"{source}: Edge {index} must be a dict, got {type(edge).__name__}"
            )

        missing_edge_fields = _check_missing_fields(
            edge, WorkflowLoader.EDGE_REQUIRED_FIELDS
        )
        if missing_edge_fields:
            raise WorkflowValidationError(
                f"{source}: Edge {index} missing required fields: {missing_edge_fields}"
            )

        source_id = edge["source"]
        target_id = edge["target"]

        if source_id not in node_ids:
            raise WorkflowValidationError(
                f"{source}: Edge {index} references unknown source node: {source_id}"
            )

        if target_id not in node_ids:
            raise WorkflowValidationError(
                f"{source}: Edge {index} references unknown target node: {target_id}"
            )

    @staticmethod
    def _validate_outputs(
        outputs: Any, node_ids: set[str], source: str | Path
    ) -> None:
        if not isinstance(outputs, list):
            raise WorkflowValidationError(
                f"{source}: 'outputs' must be a list, got {type(outputs).__name__}"
            )

        for output_id in outputs:
            if output_id not in node_ids:
                raise WorkflowValidationError(
                    f"{source}: Output references unknown node: {output_id}"
                )

    @staticmethod
    def to_executor_format(workflow: dict[str, Any]) -> dict[str, Any]:
        return {
            "nodes": workflow["nodes"],
            "edges": workflow.get("edges", []),
            "metadata": {
                "name": workflow["name"],
                "description": workflow.get("description", ""),
                "version": workflow["version"],
                "outputs": workflow.get("outputs", []),
            },
        }

    @staticmethod
    def resolve_variables(
        workflow: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Variable resolution (${node_id.output_name}) is planned but not
        # yet implemented. Returns workflow unchanged for now.
        return workflow

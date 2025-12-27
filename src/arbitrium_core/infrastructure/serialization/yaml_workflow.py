from pathlib import Path
from typing import Any

import yaml

from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class WorkflowValidationError(Exception):
    """Raised when workflow YAML validation fails."""


def _check_missing_fields(
    data: dict[str, Any], required_fields: list[str]
) -> list[str]:
    return [field for field in required_fields if field not in data]


class WorkflowLoader:
    """Load and validate YAML workflow definitions."""

    SUPPORTED_VERSIONS = ["1.0"]
    REQUIRED_FIELDS = ["version", "name", "nodes"]
    NODE_REQUIRED_FIELDS = ["id", "type"]
    EDGE_REQUIRED_FIELDS = ["source", "target"]

    @staticmethod
    def load_from_file(path: str | Path) -> dict[str, Any]:
        """Load workflow from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            dict with keys:
                - version (str): Workflow version
                - name (str): Workflow name
                - description (str): Workflow description
                - nodes (list): List of node definitions
                - edges (list): List of edge definitions
                - outputs (list): List of output node IDs

        Raises:
            WorkflowValidationError: If YAML is invalid or missing required fields
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        logger.info(f"Loading workflow from {path}")

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
            f"Loaded workflow '{data['name']}' with {len(data['nodes'])} nodes"
        )

        return data

    @staticmethod
    def load_from_string(yaml_content: str) -> dict[str, Any]:
        """Load workflow from YAML string.

        Args:
            yaml_content: YAML content as string

        Returns:
            dict with workflow definition

        Raises:
            WorkflowValidationError: If YAML is invalid
        """
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
        """Validate workflow structure.

        Args:
            data: Parsed YAML data
            source: Source file path or "<string>"

        Raises:
            WorkflowValidationError: If validation fails
        """
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

        if not isinstance(data["nodes"], list):
            raise WorkflowValidationError(
                f"{source}: 'nodes' must be a list, got {type(data['nodes']).__name__}"
            )

        if len(data["nodes"]) == 0:
            raise WorkflowValidationError(
                f"{source}: Workflow must have at least one node"
            )

        node_ids = set()
        for i, node in enumerate(data["nodes"]):
            if not isinstance(node, dict):
                raise WorkflowValidationError(
                    f"{source}: Node {i} must be a dict, got {type(node).__name__}"
                )

            missing_node_fields = _check_missing_fields(
                node, WorkflowLoader.NODE_REQUIRED_FIELDS
            )
            if missing_node_fields:
                raise WorkflowValidationError(
                    f"{source}: Node {i} missing required fields: {missing_node_fields}"
                )

            node_id = node["id"]
            if not isinstance(node_id, str) or not node_id.strip():
                raise WorkflowValidationError(
                    f"{source}: Node {i} has invalid id: {node_id}"
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

            if "properties" in node and not isinstance(
                node["properties"], dict
            ):
                raise WorkflowValidationError(
                    f"{source}: Node {node_id} properties must be a dict, "
                    f"got {type(node['properties']).__name__}"
                )

        edges = data.get("edges", [])
        if not isinstance(edges, list):
            raise WorkflowValidationError(
                f"{source}: 'edges' must be a list, got {type(edges).__name__}"
            )

        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise WorkflowValidationError(
                    f"{source}: Edge {i} must be a dict, got {type(edge).__name__}"
                )

            missing_edge_fields = _check_missing_fields(
                edge, WorkflowLoader.EDGE_REQUIRED_FIELDS
            )
            if missing_edge_fields:
                raise WorkflowValidationError(
                    f"{source}: Edge {i} missing required fields: {missing_edge_fields}"
                )

            source_id = edge["source"]
            target_id = edge["target"]

            if source_id not in node_ids:
                raise WorkflowValidationError(
                    f"{source}: Edge {i} references unknown source node: {source_id}"
                )

            if target_id not in node_ids:
                raise WorkflowValidationError(
                    f"{source}: Edge {i} references unknown target node: {target_id}"
                )

        outputs = data.get("outputs", [])
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
        """Convert YAML workflow to executor format (nodes + edges).

        Args:
            workflow: Parsed YAML workflow

        Returns:
            dict with keys:
                - nodes (list): Node definitions for executor
                - edges (list): Edge definitions for executor
                - metadata (dict): Workflow metadata (name, description, outputs)
        """
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

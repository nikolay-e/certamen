from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from certamen.domain.errors import ConfigurationError
from certamen.infrastructure.config.slim import SlimConfig, SlimModel
from certamen.infrastructure.serialization import WorkflowLoader
from certamen.shared.logging import get_contextual_logger
from certamen.workflows import resolve_workflow_path

logger = get_contextual_logger(__name__)


def materialize_workflow(slim: SlimConfig) -> dict[str, Any]:
    workflow_path = resolve_workflow_path(slim.workflow)
    logger.info("Resolved workflow '%s' to %s", slim.workflow, workflow_path)
    workflow = WorkflowLoader.load_from_file(workflow_path)
    workflow = deepcopy(workflow)

    _inject_question(workflow, slim.question)
    _inject_models(workflow, slim.models)
    _apply_overrides(workflow, slim.overrides)

    return workflow


def _inject_question(workflow: dict[str, Any], question: str) -> None:
    question_nodes = [
        n
        for n in workflow["nodes"]
        if n.get("id") == "question" and n.get("type") == "simple/text"
    ]
    if not question_nodes:
        raise ConfigurationError(
            "Workflow must contain a 'simple/text' node with id='question' "
            "to receive the slim-config question."
        )
    for node in question_nodes:
        node.setdefault("properties", {})
        node["properties"]["texts"] = [question]


def _inject_models(
    workflow: dict[str, Any], models: dict[str, SlimModel]
) -> None:
    llm_nodes = [n for n in workflow["nodes"] if n.get("type") == "simple/llm"]

    if len(models) != len(llm_nodes):
        raise ConfigurationError(
            f"Workflow '{workflow.get('name')}' expects "
            f"{len(llm_nodes)} models (one per simple/llm node), "
            f"got {len(models)} in slim config. "
            "Match the count or change workflow."
        )

    model_iter = iter(models.values())
    for node in llm_nodes:
        node.setdefault("properties", {})
        model = next(model_iter)
        node["properties"]["provider"] = model.provider
        node["properties"]["model_name"] = model.model_name
        if model.display_name is not None:
            node["properties"]["name"] = model.display_name
        if model.temperature is not None:
            node["properties"]["temperature"] = model.temperature
        if model.max_tokens is not None:
            node["properties"]["max_tokens"] = model.max_tokens
        if model.reasoning_effort is not None:
            node["properties"]["reasoning_effort"] = model.reasoning_effort


def _apply_overrides(
    workflow: dict[str, Any], overrides: dict[str, Any]
) -> None:
    if not overrides:
        return

    nodes_by_id = {n["id"]: n for n in workflow["nodes"]}
    for dotted_path, value in overrides.items():
        segments = dotted_path.split(".")
        if len(segments) < 2:
            raise ConfigurationError(
                f"Override path '{dotted_path}' must be at least "
                "node_id.property"
            )

        node_id, *property_path = segments
        node = nodes_by_id.get(node_id)
        if node is None:
            raise ConfigurationError(
                f"Override target '{dotted_path}': node '{node_id}' not "
                f"found in workflow. Known nodes: "
                f"{sorted(nodes_by_id.keys())}"
            )

        cursor: dict[str, Any] = node.setdefault("properties", {})
        for key in property_path[:-1]:
            next_cursor = cursor.get(key)
            if not isinstance(next_cursor, dict):
                raise ConfigurationError(
                    f"Override path '{dotted_path}' cannot descend into "
                    f"non-dict '{key}' on node '{node_id}'"
                )
            cursor = next_cursor
        cursor[property_path[-1]] = value
        logger.info("Applied override %s = %r", dotted_path, value)


def slim_to_certamen_settings(slim: SlimConfig) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "models": {
            key: model.model_dump() for key, model in slim.models.items()
        },
    }
    if slim.secrets is not None:
        settings["secrets"] = slim.secrets.model_dump()
    if slim.outputs_dir is not None:
        settings["outputs_dir"] = slim.outputs_dir
    if slim.logging is not None:
        settings["logging"] = slim.logging.model_dump()
    return settings


def load_and_materialize(
    config_path: str | Path,
) -> tuple[SlimConfig, dict[str, Any]]:
    from certamen.infrastructure.config.slim import load_slim_config

    slim = load_slim_config(config_path)
    workflow = materialize_workflow(slim)
    return slim, workflow

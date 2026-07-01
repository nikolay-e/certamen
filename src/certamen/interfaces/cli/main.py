#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path
from typing import Any, NoReturn

import colorama

from certamen.domain.errors import FatalError
from certamen.interfaces.cli.args import parse_arguments
from certamen.interfaces.cli.ui import (
    cli_cyan,
    cli_error,
    cli_success,
    cli_warning,
)
from certamen.shared.constants import DEFAULT_CONFIG_FILE
from certamen.shared.logging import get_contextual_logger

_USER_INTERRUPTED_MSG = "\nInterrupted by user. Exiting..."


class App:
    def __init__(self, args: dict[str, object] | None = None) -> None:
        self.args = args if args is not None else parse_arguments()
        self.logger = get_contextual_logger("certamen.cli")
        self.outputs_dir = self._get_outputs_dir()

    def _fatal_error(self, message: str) -> NoReturn:
        self.logger.error(message)
        raise FatalError(message)

    def _get_outputs_dir(self) -> str | None:
        outputs_dir_arg = self.args.get("outputs_dir")
        if outputs_dir_arg is None:
            return None
        return str(outputs_dir_arg)

    async def run(self) -> None:
        self.logger.info("Starting Certamen Framework")

        config_path = str(self.args.get("config", DEFAULT_CONFIG_FILE))
        if not _config_is_slim(config_path):
            self._fatal_error(
                f"Config '{config_path}' is not a slim config "
                "(missing top-level 'workflow:' key, or contains legacy "
                "tournament/knowledge_bank/features blocks). The legacy "
                "ModelComparison engine has been removed; migrate to the "
                "slim schema. See config.example.yml."
            )

        await self._run_slim(config_path)

    async def _run_slim(self, config_path: str) -> None:
        from certamen.application.slim_loader import load_and_materialize
        from certamen.application.workflow.nodes import register_all
        from certamen.domain.errors import ConfigurationError
        from certamen.infrastructure.secrets.env_secrets import load_secrets

        register_all()

        try:
            slim, workflow = load_and_materialize(config_path)
        except ConfigurationError as exc:
            self._fatal_error(str(exc))

        if not bool(self.args.get("no_secrets", False)) and slim.secrets:
            providers = sorted(
                {model.provider.lower() for model in slim.models.values()}
            )
            try:
                load_secrets({"secrets": slim.secrets.model_dump()}, providers)
            except Exception as exc:
                # nosemgrep: python-logger-credential-disclosure
                self.logger.warning(
                    "Failed to load secrets: %s — continuing", exc
                )

        outputs_dir = self.outputs_dir or slim.outputs_dir

        question_override = self._read_question_override()
        if question_override is not None:
            _override_question_in_workflow(workflow, question_override)

        try:
            await _execute_workflow_dict(
                workflow,
                source_label=f"slim:{config_path}",
                outputs_dir=outputs_dir,
                verbose=bool(self.args.get("verbose", False)),
                logger=self.logger,
            )
        except FatalError:
            raise
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            print(_USER_INTERRUPTED_MSG)
        except Exception as err:
            self._fatal_error(f"Error during tournament: {err!s}")

        self.logger.info("Certamen Framework completed successfully")

    def _read_question_override(self) -> str | None:
        question_path = self.args.get("question")
        if not question_path:
            return None
        path = Path(str(question_path))
        if not path.is_file():
            self._fatal_error(f"Question file not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._fatal_error(f"Failed to read question file '{path}': {exc}")
        return content.strip() or None


def _override_question_in_workflow(
    workflow: dict[str, Any], question: str
) -> None:
    for node in workflow["nodes"]:
        if node.get("id") == "question" and node.get("type") == "simple/text":
            node.setdefault("properties", {})["texts"] = [question]


_LEGACY_CONFIG_KEYS = (
    "tournament",
    "knowledge_bank",
    "features",
    "prompts",
    "retry",
    "reasoning_perspectives",
)


def _config_is_slim(config_path: str) -> bool:
    import yaml

    path = Path(config_path)
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (yaml.YAMLError, OSError):
        return False
    if not isinstance(raw, dict):
        return False
    if not isinstance(raw.get("workflow"), str):
        return False
    return not any(key in raw for key in _LEGACY_CONFIG_KEYS)


def _list_workflows() -> None:
    from pathlib import Path

    from certamen.workflows.registry import BUILTIN_WORKFLOWS

    cli_cyan("\nBuilt-in workflows:")
    for name in BUILTIN_WORKFLOWS:
        print(f"  {name}")

    user_dir = Path.home() / ".certamen" / "workflows"
    if user_dir.is_dir():
        user_workflows = sorted(p.stem for p in user_dir.glob("*.yml"))
        if user_workflows:
            cli_cyan("\nUser workflows (~/.certamen/workflows/):")
            for name in user_workflows:
                print(f"  {name}")

    total = len(BUILTIN_WORKFLOWS) + (
        len(list((Path.home() / ".certamen" / "workflows").glob("*.yml")))
        if user_dir.is_dir()
        else 0
    )
    print(f"\nTotal: {total} workflow(s)")


def _show_workflow(name_or_path: str) -> None:
    from certamen.infrastructure.serialization import (
        WorkflowLoader,
        WorkflowValidationError,
    )
    from certamen.workflows.registry import (
        BuiltinWorkflowNotFoundError,
        resolve_workflow_path,
    )

    try:
        path = resolve_workflow_path(name_or_path)
    except BuiltinWorkflowNotFoundError as exc:
        raise FatalError(str(exc)) from exc

    try:
        workflow = WorkflowLoader.load_from_file(str(path))
    except WorkflowValidationError as exc:
        raise FatalError(f"Workflow error: {exc}") from exc
    except FileNotFoundError:
        raise FatalError(f"File not found: {path}") from None

    cli_cyan(f"\nWorkflow: {workflow['name']}")
    description = workflow.get("description", "").strip()
    if description:
        print(f"Description: {description}")
    print(f"Version: {workflow.get('version', 'unknown')}")
    print(f"File: {path}")

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    outputs = workflow.get("outputs", [])

    print(f"\nNodes ({len(nodes)}):")
    node_types: dict[str, int] = {}
    for node in nodes:
        t = node.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    for node_type, count in sorted(node_types.items()):
        suffix = f" x{count}" if count > 1 else ""
        print(f"  {node_type}{suffix}")

    print(f"\nEdges: {len(edges)}")
    if outputs:
        print(f"Outputs: {', '.join(outputs)}")


def _print_node_types() -> None:
    from certamen.application.workflow.schema import list_node_types

    node_types = list_node_types()
    cli_cyan("\nAvailable node types:")
    for node_type in node_types:
        print(f"  - {node_type}")
    print(f"\nTotal: {len(node_types)} node types")


def _validate_workflow_file(file_path: str) -> None:
    from certamen.application.workflow.registry import registry
    from certamen.infrastructure.serialization import (
        WorkflowLoader,
        WorkflowValidationError,
    )

    try:
        workflow = WorkflowLoader.load_from_file(file_path)
        unknown_types = sorted(
            {
                node["type"]
                for node in workflow["nodes"]
                if registry.get(node["type"]) is None
            }
        )
        if unknown_types:
            raise FatalError(
                "Validation failed: unknown node type(s): "
                f"{', '.join(unknown_types)}"
            )
        cli_success("Workflow is valid")
        print(f"\nWorkflow: {workflow['name']}")
        print(f"Description: {workflow.get('description', 'N/A')}")
        print(f"Nodes: {len(workflow['nodes'])}")
        print(f"Edges: {len(workflow.get('edges', []))}")
    except WorkflowValidationError as e:
        raise FatalError(f"Validation failed: {e}") from e
    except FileNotFoundError:
        raise FatalError(f"File not found: {file_path}") from None


def _print_node_outputs(node_id: str, node_data: dict[str, Any]) -> None:
    cli_cyan(f"\n[{node_id}]")
    for key, value in node_data.items():
        print(f"  {key}: {value}")


def _print_workflow_outputs(
    outputs: dict[str, Any], output_nodes: list[str]
) -> None:
    if output_nodes:
        for node_id in output_nodes:
            if node_id in outputs:
                _print_node_outputs(node_id, outputs[node_id])
    else:
        for node_id, node_data in outputs.items():
            _print_node_outputs(node_id, node_data)


async def _execute_workflow_file(
    file_path: str, verbose: bool, logger: Any
) -> None:
    from certamen.application.execution.sync_executor import SyncExecutor
    from certamen.infrastructure.serialization import (
        WorkflowLoader,
        WorkflowValidationError,
    )

    try:
        workflow = WorkflowLoader.load_from_file(file_path)
        logger.info("Loaded workflow: %s", workflow["name"])

        executor_data = WorkflowLoader.to_executor_format(workflow)
        executor = SyncExecutor(verbose=verbose)

        logger.info("Validating workflow...")
        validation = executor.validate(
            executor_data["nodes"], executor_data["edges"]
        )

        if not validation["valid"]:
            errors = "; ".join(validation["errors"])
            raise FatalError(f"Workflow validation failed: {errors}")

        if validation["warnings"]:
            cli_warning("Warnings:")
            for warning in validation["warnings"]:
                print(f"  {warning}")

        logger.info("Executing workflow...")
        result = await executor.execute(
            executor_data["nodes"], executor_data["edges"]
        )

        if "error" in result:
            raise FatalError(f"Workflow execution failed: {result['error']}")

        cli_success("\n=== Workflow Results ===")
        outputs = result.get("outputs", {})
        output_nodes = executor_data["metadata"].get("outputs", [])
        _print_workflow_outputs(outputs, output_nodes)

    except WorkflowValidationError as e:
        raise FatalError(f"Workflow error: {e}") from e
    except FileNotFoundError:
        raise FatalError(f"File not found: {file_path}") from None


async def _execute_tournament_workflow(
    file_path: str,
    outputs_dir: str | None,
    verbose: bool,
    logger: Any,
) -> None:
    from certamen.infrastructure.serialization import (
        WorkflowLoader,
        WorkflowValidationError,
    )

    try:
        workflow = WorkflowLoader.load_from_file(file_path)
    except WorkflowValidationError as e:
        raise FatalError(f"Workflow error: {e}") from e
    except FileNotFoundError as exc:
        raise FatalError(f"File not found: {file_path}") from exc

    await _execute_workflow_dict(
        workflow,
        source_label=file_path,
        outputs_dir=outputs_dir,
        verbose=verbose,
        logger=logger,
    )


def _extract_workflow_question(workflow: dict[str, Any]) -> str | None:
    for node in workflow.get("nodes", []):
        if node.get("id") == "question" and node.get("type") == "simple/text":
            texts = node.get("properties", {}).get("texts", [])
            if texts:
                return str(texts[0])
    return None


def _extract_workflow_model_keys(workflow: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for node in workflow.get("nodes", []):
        if node.get("type") == "simple/llm":
            props = node.get("properties", {})
            name = props.get("name") or node.get("id") or ""
            if name:
                keys.append(str(name))
    return keys


def _extract_champion_from_outputs(
    outputs: dict[str, Any],
) -> dict[str, Any] | None:
    for value in outputs.values():
        if not isinstance(value, dict):
            continue
        champion = value.get("champion")
        if isinstance(champion, dict):
            model = champion.get("model")
            if isinstance(model, dict):
                return {
                    "name": model.get("name")
                    or model.get("display_name")
                    or model.get("model_name"),
                    "model_name": model.get("model_name"),
                    "provider": model.get("provider"),
                }
    return None


async def _execute_workflow_dict(
    workflow: dict[str, Any],
    source_label: str,
    outputs_dir: str | None,
    verbose: bool,
    logger: Any,
) -> None:
    import json
    from pathlib import Path

    from certamen.application.execution.async_executor import AsyncExecutor
    from certamen.infrastructure.events import (
        JsonlEventHandler,
        generate_run_id,
    )
    from certamen.infrastructure.serialization import WorkflowLoader

    logger.info("Loaded workflow: %s", workflow.get("name", source_label))
    executor_data = WorkflowLoader.to_executor_format(workflow)

    base_dir = Path(outputs_dir) if outputs_dir else Path("reports")
    run_id = generate_run_id()
    run_dir = base_dir / "runs" / run_id

    with JsonlEventHandler(run_dir, run_id) as event_handler:
        question = _extract_workflow_question(workflow)
        model_keys = _extract_workflow_model_keys(workflow)
        event_handler.publish(
            "tournament_started",
            {
                "run_id": run_id,
                "workflow_name": workflow.get("name", source_label),
                "workflow_source": source_label,
                "node_count": len(executor_data["nodes"]),
                "edge_count": len(executor_data["edges"]),
                "question": question,
                "models": model_keys,
            },
        )

        async def broadcast_event(  # awaited by AsyncExecutor._broadcast
            message_str: str,
        ) -> None:
            try:
                msg = json.loads(message_str)
            except (json.JSONDecodeError, TypeError):
                return
            event_handler.publish(
                msg.get("type", "node_event"),
                {
                    "node_id": msg.get("node_id"),
                    "data": msg.get("data"),
                },
            )

        executor = AsyncExecutor(broadcast_fn=broadcast_event)
        validation = executor.validate(
            executor_data["nodes"], executor_data["edges"]
        )
        if not validation["valid"]:
            errors = "; ".join(validation["errors"])
            event_handler.publish(
                "tournament_ended",
                {"error": f"Validation failed: {errors}"},
            )
            raise FatalError(f"Workflow validation failed: {errors}")

        if validation.get("warnings"):
            cli_warning("Warnings:")
            for warning in validation["warnings"]:
                print(f"  {warning}")

        cli_cyan(f"\n=== Tournament Run: {run_id} ===")
        if verbose:
            print(f"Output dir: {run_dir}")

        result = await executor.execute(
            executor_data["nodes"], executor_data["edges"]
        )

        outputs = result.get("outputs", {})
        output_nodes = executor_data["metadata"].get("outputs", [])

        if "error" in result:
            event_handler.publish(
                "tournament_ended", {"error": result["error"]}
            )
            raise FatalError(f"Workflow execution failed: {result['error']}")

        champion_info = _extract_champion_from_outputs(outputs)
        event_handler.publish(
            "tournament_ended",
            {
                "run_id": run_id,
                "completed_nodes": len(outputs),
                "output_nodes": output_nodes,
                "champion": champion_info,
            },
        )

        cli_success("\n=== Workflow Results ===")
        _print_workflow_outputs(outputs, output_nodes)
        print(f"\nEvents log: {run_dir / 'events.jsonl'}")


async def run_workflow(args: dict[str, object]) -> None:
    from certamen.shared.logging import get_contextual_logger

    logger = get_contextual_logger("certamen.workflow")
    workflow_command = args.get("workflow_command")

    if workflow_command in ("list-nodes", "nodes"):
        _print_node_types()
        return

    if workflow_command in ("list", "workflows"):
        _list_workflows()
        return

    if workflow_command in ("show", "inspect"):
        name_or_path = str(args.get("name_or_path", ""))
        if not name_or_path:
            raise FatalError("No workflow name or path specified")
        _show_workflow(name_or_path)
        return

    file_path = str(args.get("file", ""))
    if not file_path:
        raise FatalError("No file specified for workflow command")

    if workflow_command in ("validate", "check"):
        _validate_workflow_file(file_path)
        return

    if workflow_command in ("execute", "run", "exec"):
        await _execute_workflow_file(
            file_path, bool(args.get("verbose", True)), logger
        )
        return

    raise FatalError(f"Unknown workflow command: {workflow_command}")


def _run_gui_command(args: dict[str, Any]) -> None:
    from certamen.interfaces.web.server import run_gui_server

    host = str(args.get("host", "0.0.0.0"))  # noqa: S104
    port = int(args.get("port", 8765))  # type: ignore[arg-type]
    try:
        asyncio.run(run_gui_server(host=host, port=port))
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


def _run_render_command(args: dict[str, Any]) -> None:
    from certamen.interfaces.render import write_run_report

    raw = str(args.get("run_dir", ""))
    outputs_dir = Path(str(args.get("outputs_dir", "outputs")))
    run_path = Path(raw)
    if not run_path.is_dir():
        candidate = outputs_dir / "runs" / raw
        if candidate.is_dir():
            run_path = candidate
        else:
            cli_error(f"Run directory not found: {raw}")
            sys.exit(1)

    out = args.get("output")
    out_path = Path(str(out)) if out else None
    try:
        written = write_run_report(run_path, out_path)
    except Exception as e:
        cli_error(f"Render failed: {e}")
        sys.exit(1)
    print(f"Wrote report: {written}")


def _run_workflow_command(args: dict[str, Any]) -> None:
    from certamen.application.workflow.nodes import register_all

    register_all()
    try:
        asyncio.run(run_workflow(args))
    except FatalError as e:
        cli_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print(_USER_INTERRUPTED_MSG)
        sys.exit(130)


def _run_tournament_workflow_command(args: dict[str, Any]) -> None:
    from certamen.application.workflow.nodes import register_all
    from certamen.shared.logging import get_contextual_logger as _get_logger

    register_all()
    logger = _get_logger("certamen.tournament_workflow")
    tw_outputs = args.get("outputs_dir")
    outputs_dir_str = str(tw_outputs) if tw_outputs is not None else None
    try:
        asyncio.run(
            _execute_tournament_workflow(
                str(args.get("workflow")),
                outputs_dir_str,
                bool(args.get("verbose", False)),
                logger,
            )
        )
    except FatalError as e:
        cli_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print(_USER_INTERRUPTED_MSG)
        sys.exit(130)


def run_from_cli() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from certamen.shared.logging import setup_logging

    args = parse_arguments()

    setup_logging(
        debug=bool(args.get("debug", False)),
        verbose=bool(args.get("verbose", False)),
        log_dir="reports/logs",
    )

    from certamen.interfaces.cli.ui import configure_display

    configure_display(use_color=not bool(args.get("no_color", False)))

    colorama.init(autoreset=True)

    command = args.get("command", "tournament")

    if command in ("gui", "web"):
        _run_gui_command(args)
        return

    if command == "render":
        _run_render_command(args)
        return

    if command == "workflow":
        _run_workflow_command(args)
        return

    if args.get("workflow"):
        _run_tournament_workflow_command(args)
        return

    try:
        app = App(args)
        asyncio.run(app.run())
    except FatalError:
        sys.exit(1)
    except KeyboardInterrupt:
        print(_USER_INTERRUPTED_MSG)
        sys.exit(130)


if __name__ == "__main__":
    run_from_cli()

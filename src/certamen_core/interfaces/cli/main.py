#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import colorama

from certamen_core import Certamen
from certamen_core.domain.errors import FatalError
from certamen_core.interfaces.cli.args import parse_arguments
from certamen_core.interfaces.cli.input import async_input
from certamen_core.interfaces.cli.ui import (
    cli_cyan,
    cli_error,
    cli_success,
    cli_warning,
)
from certamen_core.shared.constants import DEFAULT_CONFIG_FILE
from certamen_core.shared.logging import get_contextual_logger

if TYPE_CHECKING:
    from certamen_core.infrastructure.config.loader import Config

CERTAMEN_NOT_INITIALIZED_MSG = "Certamen not initialized"
_USER_INTERRUPTED_MSG = "\nInterrupted by user. Exiting..."


class App:
    def __init__(self, args: dict[str, object] | None = None) -> None:
        self.args = args if args is not None else parse_arguments()
        self.logger = get_contextual_logger("certamen.cli")
        self.outputs_dir = self._get_outputs_dir()
        self.certamen: Certamen | None = None

    def _fatal_error(self, message: str) -> NoReturn:
        self.logger.error(message)
        raise FatalError(message)

    def _get_outputs_dir(self) -> str | None:
        outputs_dir_arg = self.args.get("outputs_dir")
        if outputs_dir_arg is None:
            return None
        return str(outputs_dir_arg)

    def _load_config(self, config_path: str) -> "Config":
        from certamen_core.infrastructure.config.loader import Config

        config_obj = Config(config_path)
        if config_obj.load():
            return config_obj

        self._fatal_error(
            f"Failed to load configuration from '{config_path}'. "
            f"Please ensure the file exists and is valid YAML. "
            f"Use --config to specify a different config file."
        )

    async def _try_create_certamen_from_config_obj(
        self, config_obj: "Config", skip_secrets: bool
    ) -> Certamen:
        if self.outputs_dir is not None:
            config_obj.config_data["outputs_dir"] = self.outputs_dir
        return await Certamen.from_settings(
            settings=config_obj.config_data,
            skip_secrets=skip_secrets,
        )

    async def _create_certamen_from_config(
        self, config_path: str, skip_secrets: bool
    ) -> Certamen:
        config_obj = self._load_config(config_path)

        return await self._try_create_certamen_from_config_obj(
            config_obj, skip_secrets
        )

    def _filter_requested_models(self) -> None:
        if not self.args.get("models") or self.certamen is None:
            return

        models_arg: str = self.args.get("models")  # type: ignore[assignment]
        requested_models = [m.strip() for m in models_arg.split(",")]

        filtered_models = {
            key: model
            for key, model in self.certamen.healthy_models.items()
            if key in requested_models
        }

        if not filtered_models:
            self._fatal_error(
                f"None of the requested models ({', '.join(requested_models)}) are available or healthy"
            )

        self.logger.info(
            "Filtering to requested models: %s",
            ", ".join(filtered_models.keys()),
        )
        self.certamen._healthy_models = filtered_models

    def _validate_certamen_ready(self) -> None:
        if self.certamen is None:
            self._fatal_error(CERTAMEN_NOT_INITIALIZED_MSG)
        assert self.certamen is not None
        if not self.certamen.is_ready:
            self._fatal_error("No models passed health check")

    def _reconfigure_logging_from_config(self) -> None:
        if self.certamen is None:
            return

        from certamen_core.shared.logging import setup_logging

        log_dir = self.outputs_dir
        if log_dir is None:
            try:
                log_dir = self.certamen.config.config_data.get("outputs_dir")
            except (AttributeError, TypeError):
                pass
        setup_logging(
            debug=bool(self.args.get("debug", False)),
            verbose=bool(self.args.get("verbose", False)),
            log_dir=str(log_dir) if log_dir else None,
        )

    async def _initialize_certamen(self) -> None:
        config_path = str(self.args.get("config", DEFAULT_CONFIG_FILE))
        skip_secrets = bool(self.args.get("no_secrets", False))

        self.logger.info("Loading configuration from %s", config_path)

        self.certamen = await self._create_certamen_from_config(
            config_path, skip_secrets
        )

        self._reconfigure_logging_from_config()

        self._filter_requested_models()

        self._validate_certamen_ready()

    async def _get_app_question(self) -> str:
        if self.certamen is None:
            self._fatal_error(CERTAMEN_NOT_INITIALIZED_MSG)

        question = ""

        if self.args.get("interactive", False):
            self.logger.info(
                "Enter your question:", extra={"display_type": "header"}
            )
            question = await async_input("> ")
            return question.strip()

        question_path = self.args.get("question")
        if question_path:
            path = Path(str(question_path))
            if not path.exists() or not path.is_file():
                self._fatal_error(f"Question file not found: {path}")
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                self._fatal_error(
                    f"Failed to read question file '{path}': {e!s}"
                )
            if content.strip():
                self.logger.info("Using question from file: %s", path)
                return content.strip()

        assert self.certamen is not None
        config_question = self.certamen.config_data.get("question")
        if config_question:
            self.logger.info("Using question from config file")
            return str(config_question).strip()

        self.logger.info(
            "No question file or config question found, entering interactive mode"
        )
        self.logger.info(
            "Enter your question:", extra={"display_type": "header"}
        )
        question = await async_input("> ")

        return question.strip()

    async def run(self) -> None:
        self.logger.info("Starting Certamen Framework")

        await self._initialize_certamen()

        if self.certamen is None:
            self._fatal_error(CERTAMEN_NOT_INITIALIZED_MSG)

        question = await self._get_app_question()

        assert self.certamen is not None
        try:
            _result, _metrics = await self.certamen.run_tournament(question)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            print(_USER_INTERRUPTED_MSG)
        except Exception as err:
            self._fatal_error(f"Error during tournament: {err!s}")

        self.logger.info("Certamen Framework completed successfully")


def _print_node_types() -> None:
    from certamen_core.application.workflow.schema import list_node_types

    node_types = list_node_types()
    cli_cyan("\nAvailable node types:")
    for node_type in node_types:
        print(f"  - {node_type}")
    print(f"\nTotal: {len(node_types)} node types")


def _validate_workflow_file(file_path: str) -> None:
    from certamen_core.infrastructure.serialization import (
        WorkflowLoader,
        WorkflowValidationError,
    )

    try:
        workflow = WorkflowLoader.load_from_file(file_path)
        cli_success("Workflow is valid")
        print(f"\nWorkflow: {workflow['name']}")
        print(f"Description: {workflow.get('description', 'N/A')}")
        print(f"Nodes: {len(workflow['nodes'])}")
        print(f"Edges: {len(workflow.get('edges', []))}")
    except WorkflowValidationError as e:
        raise FatalError(f"Validation failed: {e}") from e
    except FileNotFoundError:
        raise FatalError(f"File not found: {file_path}") from None


def _print_workflow_outputs(
    outputs: dict[str, Any], output_nodes: list[str]
) -> None:
    if output_nodes:
        for node_id in output_nodes:
            if node_id in outputs:
                cli_cyan(f"\n[{node_id}]")
                for key, value in outputs[node_id].items():
                    print(f"  {key}: {value}")
    else:
        for node_id, node_outputs in outputs.items():
            cli_cyan(f"\n[{node_id}]")
            for key, value in node_outputs.items():
                print(f"  {key}: {value}")


async def _execute_workflow_file(
    file_path: str, verbose: bool, logger: Any
) -> None:
    from certamen_core.application.execution.sync_executor import SyncExecutor
    from certamen_core.infrastructure.serialization import (
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


async def run_workflow(args: dict[str, object]) -> None:
    from certamen_core.shared.logging import get_contextual_logger

    logger = get_contextual_logger("certamen.workflow")
    workflow_command = args.get("workflow_command")

    if workflow_command in ("list-nodes", "nodes"):
        _print_node_types()
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


def run_from_cli() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from certamen_core.shared.logging import setup_logging

    args = parse_arguments()

    setup_logging(
        debug=bool(args.get("debug", False)),
        verbose=bool(args.get("verbose", False)),
    )

    from certamen_core.interfaces.cli.ui import configure_display

    configure_display(use_color=not bool(args.get("no_color", False)))

    colorama.init(autoreset=True)

    command = args.get("command", "tournament")

    if command == "workflow":
        from certamen_core.application.workflow.nodes import register_all

        register_all()

        try:
            asyncio.run(run_workflow(args))
        except FatalError as e:
            cli_error(str(e))
            sys.exit(1)
        except KeyboardInterrupt:
            print(_USER_INTERRUPTED_MSG)
            sys.exit(130)
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

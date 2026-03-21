#!/usr/bin/env python3
"""
Certamen Framework - LLM Comparison and Evaluation Tool

This is the main entry point for the Certamen Framework CLI application.
"""

import asyncio
import sys
from typing import TYPE_CHECKING

import colorama

# Public API imports - CLI uses only exported interface
from certamen import Certamen

# CLI-specific components
from certamen.cli.args import parse_arguments
from certamen.logging import get_contextual_logger
from certamen.utils.async_ import async_input
from certamen.utils.exceptions import FatalError

if TYPE_CHECKING:
    from certamen.config.loader import Config

# Constants for configuration
DEFAULT_CONFIG_FILE = "config.example.yml"
CERTAMEN_NOT_INITIALIZED_MSG = "Certamen not initialized"


class App:
    def __init__(self, args: dict[str, object] | None = None) -> None:
        self.args = args if args is not None else parse_arguments()
        self.logger = get_contextual_logger("certamen.cli")
        self.outputs_dir = self._get_outputs_dir()
        self.certamen: Certamen | None = None

    def _fatal_error(self, message: str) -> None:
        self.logger.error(message)
        raise FatalError(message)

    def _get_outputs_dir(self) -> str | None:
        outputs_dir_arg = self.args.get("outputs_dir")
        if outputs_dir_arg is None:
            return None
        return str(outputs_dir_arg)

    def _load_config(self, config_path: str) -> "Config":
        from certamen.config.loader import Config

        config_obj = Config(config_path)
        if config_obj.load():
            return config_obj

        # Config file not found or invalid - fail with clear error
        self._fatal_error(
            f"Failed to load configuration from '{config_path}'. "
            f"Please ensure the file exists and is valid YAML. "
            f"Use --config to specify a different config file."
        )

        # This line is never reached due to _fatal_error raising exception
        return config_obj

    async def _try_create_certamen_from_config_obj(
        self, config_obj: "Config", skip_secrets: bool
    ) -> Certamen:
        """Try to create Certamen instance from a config object."""
        # Only override outputs_dir if explicitly provided via CLI
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
            f"Filtering to requested models: {', '.join(filtered_models.keys())}"
        )
        self.certamen._healthy_models = filtered_models

    def _validate_certamen_ready(self) -> None:
        """Validate that Certamen has healthy models."""
        if self.certamen is None:
            self._fatal_error(CERTAMEN_NOT_INITIALIZED_MSG)
        assert self.certamen is not None
        if not self.certamen.is_ready:
            self._fatal_error("❌ No models passed health check")

    def _reconfigure_logging_from_config(self) -> None:
        if self.certamen is None:
            return

        from certamen.logging import setup_logging

        # Reconfigure logging with config settings
        # Note: File logging always uses JSON format (standard behavior)
        setup_logging(
            debug=bool(self.args.get("debug", False)),
            verbose=bool(self.args.get("verbose", False)),
        )

    async def _initialize_certamen(self) -> None:
        config_path = str(self.args.get("config", DEFAULT_CONFIG_FILE))
        skip_secrets = bool(self.args.get("no_secrets", False))

        self.logger.info(f"Loading configuration from {config_path}")

        # Create Certamen from config with fallback support
        self.certamen = await self._create_certamen_from_config(
            config_path, skip_secrets
        )

        # Reconfigure logging with settings from config
        self._reconfigure_logging_from_config()

        # Filter models if specific models were requested
        self._filter_requested_models()

        # Validate that we have healthy models
        self._validate_certamen_ready()

    async def _get_app_question(self) -> str:
        if self.certamen is None:
            self._fatal_error(CERTAMEN_NOT_INITIALIZED_MSG)

        question = ""

        # Check if interactive mode is requested
        if self.args.get("interactive", False):
            self.logger.info(
                "Enter your question:", extra={"display_type": "header"}
            )
            question = await async_input("> ")
            return question.strip()

        assert self.certamen is not None
        config_question = self.certamen.config_data.get("question")
        if config_question:
            self.logger.info("Using question from config file")
            return str(config_question).strip()

        # No question in config, fall back to interactive mode
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

        # Initialize certamen
        await self._initialize_certamen()

        if self.certamen is None:
            self._fatal_error(CERTAMEN_NOT_INITIALIZED_MSG)

        # Get the question
        question = await self._get_app_question()

        # Run tournament
        assert self.certamen is not None
        try:
            _result, _metrics = await self.certamen.run_tournament(question)
            # Result is displayed via logging during tournament execution
            # Metrics are also logged by the tournament itself
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            print("\nInterrupted by user. Exiting...")
        except Exception as err:
            self._fatal_error(f"Error during tournament: {err!s}")

        self.logger.info("Certamen Framework completed successfully")


def run_gui(args: dict[str, object]) -> None:
    from certamen.gui.server import GUIServer

    host = str(args.get("host", "0.0.0.0"))
    port = int(args.get("port", 8765))  # type: ignore[call-overload]

    print(f"Starting Certamen GUI server at http://{host}:{port}")
    print(f"WebSocket endpoint: ws://{host}:{port}/ws")
    print("Press Ctrl+C to stop")

    server = GUIServer(host=host, port=port)
    server.run()


async def run_workflow(args: dict[str, object]) -> None:
    """Execute YAML workflow commands."""
    from certamen.logging import get_contextual_logger
    from certamen.serialization import WorkflowLoader, WorkflowValidationError
    from certamen_core.executor import SyncExecutor

    logger = get_contextual_logger("certamen.workflow")
    workflow_command = args.get("workflow_command")

    if workflow_command == "list-nodes" or workflow_command == "nodes":
        node_types = WorkflowLoader.list_node_types()
        print(
            f"\n{colorama.Fore.CYAN}Available node types:{colorama.Style.RESET_ALL}"
        )
        for node_type in node_types:
            print(f"  - {node_type}")
        print(f"\nTotal: {len(node_types)} node types")
        return

    file_path = str(args.get("file", ""))
    if not file_path:
        print(
            f"{colorama.Fore.RED}Error: No file specified{colorama.Style.RESET_ALL}"
        )
        sys.exit(1)

    if workflow_command == "validate" or workflow_command == "check":
        try:
            workflow = WorkflowLoader.load_from_file(file_path)
            print(
                f"{colorama.Fore.GREEN}✓ Workflow is valid{colorama.Style.RESET_ALL}"
            )
            print(f"\nWorkflow: {workflow['name']}")
            print(f"Description: {workflow.get('description', 'N/A')}")
            print(f"Nodes: {len(workflow['nodes'])}")
            print(f"Edges: {len(workflow.get('edges', []))}")
        except WorkflowValidationError as e:
            print(
                f"{colorama.Fore.RED}✗ Validation failed:{colorama.Style.RESET_ALL}"
            )
            print(f"  {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(
                f"{colorama.Fore.RED}✗ File not found: {file_path}{colorama.Style.RESET_ALL}"
            )
            sys.exit(1)
        return

    if workflow_command in ("execute", "run", "exec"):
        try:
            workflow = WorkflowLoader.load_from_file(file_path)
            logger.info(f"Loaded workflow: {workflow['name']}")

            executor_data = WorkflowLoader.to_executor_format(workflow)
            executor = SyncExecutor(verbose=bool(args.get("verbose", True)))

            logger.info("Validating workflow...")
            validation = executor.validate(
                executor_data["nodes"], executor_data["edges"]
            )

            if not validation["valid"]:
                print(
                    f"{colorama.Fore.RED}✗ Workflow validation failed:{colorama.Style.RESET_ALL}"
                )
                for error in validation["errors"]:
                    print(f"  Error: {error}")
                sys.exit(1)

            if validation["warnings"]:
                print(
                    f"{colorama.Fore.YELLOW}Warnings:{colorama.Style.RESET_ALL}"
                )
                for warning in validation["warnings"]:
                    print(f"  {warning}")

            logger.info("Executing workflow...")
            result = await executor.execute(
                executor_data["nodes"], executor_data["edges"]
            )

            if "error" in result:
                print(
                    f"{colorama.Fore.RED}✗ Execution failed:{colorama.Style.RESET_ALL}"
                )
                print(f"  {result['error']}")
                sys.exit(1)

            print(
                f"\n{colorama.Fore.GREEN}=== Workflow Results ==={colorama.Style.RESET_ALL}"
            )
            outputs = result.get("outputs", {})
            output_nodes = executor_data["metadata"].get("outputs", [])

            if output_nodes:
                for node_id in output_nodes:
                    if node_id in outputs:
                        print(
                            f"\n{colorama.Fore.CYAN}[{node_id}]{colorama.Style.RESET_ALL}"
                        )
                        for key, value in outputs[node_id].items():
                            print(f"  {key}: {value}")
            else:
                for node_id, node_outputs in outputs.items():
                    print(
                        f"\n{colorama.Fore.CYAN}[{node_id}]{colorama.Style.RESET_ALL}"
                    )
                    for key, value in node_outputs.items():
                        print(f"  {key}: {value}")

        except WorkflowValidationError as e:
            print(
                f"{colorama.Fore.RED}✗ Workflow error:{colorama.Style.RESET_ALL}"
            )
            print(f"  {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(
                f"{colorama.Fore.RED}✗ File not found: {file_path}{colorama.Style.RESET_ALL}"
            )
            sys.exit(1)
        return

    print(
        f"{colorama.Fore.RED}Unknown workflow command: {workflow_command}{colorama.Style.RESET_ALL}"
    )
    sys.exit(1)


def run_from_cli() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from certamen.logging import setup_logging

    args = parse_arguments()

    setup_logging(
        debug=args.get("debug", False),
        verbose=args.get("verbose", False),
    )

    colorama.init(autoreset=True)

    command = args.get("command", "tournament")

    if command == "gui":
        try:
            run_gui(args)
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)
        return

    if command == "workflow":
        # Import all nodes to register them
        import certamen_core.nodes.evaluation
        import certamen_core.nodes.flow
        import certamen_core.nodes.generation
        import certamen_core.nodes.input
        import certamen_core.nodes.knowledge
        import certamen_core.nodes.llm
        import certamen_core.nodes.output  # noqa: F401

        try:
            asyncio.run(run_workflow(args))
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            sys.exit(130)
        return

    try:
        app = App(args)
        asyncio.run(app.run())
    except FatalError:
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(130)


if __name__ == "__main__":
    run_from_cli()

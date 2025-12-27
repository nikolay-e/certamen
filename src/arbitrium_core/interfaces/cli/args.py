import argparse
import sys
from typing import Any

from arbitrium_core.__about__ import __version__
from arbitrium_core.shared.constants import DEFAULT_CONFIG_FILE


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable detailed debug logging",
    )
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose output", action="store_true"
    )


def _add_tournament_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        help="Comma-separated list of model keys to run",
    )
    parser.add_argument(
        "-c",
        "--config",
        help=f"Path to config file (CLI default: {DEFAULT_CONFIG_FILE})",
        default=DEFAULT_CONFIG_FILE,
    )
    parser.add_argument("-q", "--question", help="Path to question file")
    parser.add_argument(
        "-o",
        "--outputs-dir",
        default=None,
        help="Output directory for all files (default: current directory)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Run in interactive mode",
        action="store_true",
    )
    parser.add_argument(
        "--no-color", help="Disable colored output", action="store_true"
    )
    parser.add_argument(
        "--no-secrets", help="Skip loading secrets", action="store_true"
    )


def _add_workflow_execute_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "file",
        type=str,
        help="Path to YAML workflow file",
    )
    parser.add_argument(
        "-o",
        "--outputs-dir",
        default=None,
        help="Output directory for results (default: current directory)",
    )


def _add_workflow_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "file",
        type=str,
        help="Path to YAML workflow file to validate",
    )


def parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Arbitrium Framework - LLM Comparison and Evaluation Tool"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Arbitrium Framework {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    tournament_parser = subparsers.add_parser(
        "tournament",
        aliases=["run"],
        help="Run a tournament (default)",
    )
    _add_common_args(tournament_parser)
    _add_tournament_args(tournament_parser)

    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Execute and manage YAML workflows",
    )
    workflow_subparsers = workflow_parser.add_subparsers(
        dest="workflow_command", help="Workflow commands"
    )

    execute_parser = workflow_subparsers.add_parser(
        "execute",
        aliases=["run", "exec"],
        help="Execute a YAML workflow file",
    )
    _add_common_args(execute_parser)
    _add_workflow_execute_args(execute_parser)

    validate_parser = workflow_subparsers.add_parser(
        "validate",
        aliases=["check"],
        help="Validate a YAML workflow file",
    )
    _add_common_args(validate_parser)
    _add_workflow_validate_args(validate_parser)

    list_nodes_parser = workflow_subparsers.add_parser(
        "list-nodes",
        aliases=["nodes", "ls"],
        help="List all available node types",
    )
    _add_common_args(list_nodes_parser)

    if len(sys.argv) == 1:
        args_dict = {"command": "tournament"}
        args_dict.update(vars(tournament_parser.parse_args([])))
        return args_dict

    first_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if first_arg.startswith("-") and first_arg not in (
        "--version",
        "--help",
        "-h",
    ):
        _add_common_args(parser)
        _add_tournament_args(parser)
        args = parser.parse_args()
        args_dict = vars(args)
        args_dict["command"] = "tournament"
    else:
        args = parser.parse_args()
        args_dict = vars(args)
        if args_dict.get("command") is None:
            args_dict["command"] = "tournament"
            args_dict.update(vars(tournament_parser.parse_args([])))

    if args_dict.get("debug"):
        args_dict["verbose"] = True  # type: ignore[assignment]

    return args_dict

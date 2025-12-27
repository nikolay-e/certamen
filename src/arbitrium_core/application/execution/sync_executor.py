# pylint: disable=duplicate-code
import traceback
from typing import Any

from arbitrium_core.application.execution.executor import BaseExecutor
from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
)
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class SyncExecutor(BaseExecutor):
    """Synchronous workflow executor for CLI mode.

    Differences from AsyncExecutor:
    - No WebSocket broadcasting (prints to console instead)
    - Blocking execution (suitable for CLI scripts)
    - Simplified progress reporting
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        verbose: bool = True,
    ):
        super().__init__(config)
        self.verbose = verbose

    def _print_progress(self, message: str) -> None:
        """Print progress to console if verbose mode is enabled."""
        if self.verbose:
            print(f"[Arbitrium] {message}")

    def _initialize_execution_state(
        self,
        execution_id: str,
    ) -> tuple[ExecutionContext, dict[str, dict[str, Any]]]:
        context = ExecutionContext(
            execution_id=execution_id,
            broadcast=None,
            models={},
            config=self.config,
        )

        node_outputs: dict[str, dict[str, Any]] = {}

        return context, node_outputs

    async def _execute_single_node(
        self,
        node_id: str,
        node: BaseNode,
        inputs: dict[str, Any],
        context: ExecutionContext,
        execution_id: str,
    ) -> dict[str, Any]:
        self._log_node_start(execution_id, node_id, node.NODE_TYPE, inputs)
        self._print_progress(f"Executing node [{node_id}] ({node.NODE_TYPE})")

        missing_inputs = self._validate_required_inputs(node, inputs)
        if missing_inputs:
            self._log_missing_inputs(execution_id, node_id, missing_inputs)
            self._print_progress(
                f"Warning: Node [{node_id}] missing required inputs: {missing_inputs}"
            )

        try:
            outputs = await self._execute_node_with_timeout(
                node, inputs, context
            )

            self._log_node_complete(execution_id, node_id, outputs)
            self._print_progress(f"Node [{node_id}] completed successfully")

            return outputs

        except Exception as e:
            self._log_node_error(execution_id, node_id, node.NODE_TYPE, e)
            self._handle_node_error(e, node_id, node, execution_id)
            raise

    def _handle_node_error(
        self,
        error: Exception,
        node_id: str,
        node: BaseNode,
        execution_id: str,
    ) -> None:
        error_msg = f"Node [{node_id}] failed: {type(error).__name__}: {error}"
        self._print_progress(f"ERROR: {error_msg}")

    def _report_execution_start(
        self,
        execution_id: str,
        num_nodes: int,
        num_edges: int,
    ) -> None:
        self._print_progress(
            f"Starting workflow execution ({num_nodes} nodes, {num_edges} edges)"
        )

    def _report_layer_start(
        self,
        execution_id: str,
        layer_index: int,
        total_layers: int,
        layer: list[str],
    ) -> None:
        self._print_progress(
            f"Layer {layer_index + 1}/{total_layers}: {len(layer)} node(s)"
        )

    def _report_execution_complete(
        self,
        execution_id: str,
        num_outputs: int,
    ) -> None:
        self._print_progress(
            f"Workflow completed successfully ({num_outputs} nodes executed)"
        )

    def _handle_execution_error(
        self,
        error: Exception,
        execution_id: str,
    ) -> dict[str, Any]:
        full_traceback = traceback.format_exc()
        logger.exception(
            "Execution %s failed: %s: %s\n%s",
            execution_id,
            type(error).__name__,
            error,
            full_traceback,
        )

        error_msg = f"Execution failed: {type(error).__name__}: {error}"
        self._print_progress(f"ERROR: {error_msg}")

        return {
            "execution_id": execution_id,
            "error": str(error),
        }

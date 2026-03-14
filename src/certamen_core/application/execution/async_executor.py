# pylint: disable=duplicate-code
import asyncio
import json
import traceback
from typing import Any

from certamen_core.application.execution.executor import BaseExecutor
from certamen_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
)
from certamen_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class ExecutionCancelledError(Exception):
    pass


DEFAULT_GLOBAL_EXECUTION_TIMEOUT = 3600  # 1 hour default


class AsyncExecutor(BaseExecutor):
    def __init__(
        self,
        broadcast_fn: Any = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.broadcast_fn = broadcast_fn
        self._current_execution_id: str | None = None
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Global execution timeout (prevents runaway executions)
        self.global_timeout = (config or {}).get(
            "global_timeout", DEFAULT_GLOBAL_EXECUTION_TIMEOUT
        )

    def cancel(self) -> bool:
        if self._current_execution_id:
            logger.info(
                "Cancellation requested for execution %s",
                self._current_execution_id[:8],
            )
            self._cancel_event.set()
            return True
        return False

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise ExecutionCancelledError("Execution cancelled by user")

    async def _broadcast(self, message: dict[str, Any]) -> None:
        if self.broadcast_fn:
            await self.broadcast_fn(json.dumps(message))

    def _schedule_broadcast(self, message: dict[str, Any]) -> None:
        async def _safe_broadcast() -> None:
            try:
                await self._broadcast(message)
            except Exception as e:
                logger.warning(
                    "Failed to broadcast %s event: %s",
                    message.get("type", "unknown"),
                    e,
                )

        task = asyncio.create_task(_safe_broadcast())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _initialize_execution_state(
        self,
        execution_id: str,
    ) -> tuple[ExecutionContext, dict[str, dict[str, Any]]]:
        context = ExecutionContext(
            execution_id=execution_id,
            broadcast=self._broadcast,
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
        self._check_cancelled()

        self._log_node_start(execution_id, node_id, node.NODE_TYPE, inputs)

        await self._broadcast(
            {
                "type": "node_start",
                "execution_id": execution_id,
                "node_id": node_id,
                "node_type": node.NODE_TYPE,
            }
        )

        missing_inputs = self._validate_required_inputs(node, inputs)
        if missing_inputs:
            self._log_missing_inputs(execution_id, node_id, missing_inputs)
            await self._broadcast(
                {
                    "type": "node_warning",
                    "execution_id": execution_id,
                    "node_id": node_id,
                    "warning": f"Missing required inputs: {missing_inputs}",
                }
            )

        try:
            outputs = await self._execute_node_with_timeout(
                node, inputs, context
            )

            self._log_node_complete(execution_id, node_id, outputs)

            await self._broadcast(
                {
                    "type": "node_complete",
                    "execution_id": execution_id,
                    "node_id": node_id,
                    "outputs": outputs,
                }
            )

            return outputs

        except Exception as e:
            self._log_node_error(execution_id, node_id, node.NODE_TYPE, e)
            await self._handle_node_error(e, node_id, node, execution_id)
            raise

    async def _handle_node_error(
        self,
        error: Exception,
        node_id: str,
        node: BaseNode,
        execution_id: str,
    ) -> None:
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "node_id": node_id,
            "node_type": node.NODE_TYPE,
        }

        await self._broadcast(
            {
                "type": "node_error",
                "execution_id": execution_id,
                "node_id": node_id,
                "error": error_info,
            }
        )

    def _report_execution_start(
        self,
        execution_id: str,
        num_nodes: int,
        num_edges: int,
    ) -> None:
        self._schedule_broadcast(
            {
                "type": "execution_start",
                "execution_id": execution_id,
            }
        )

    def _report_layer_start(
        self,
        execution_id: str,
        layer_index: int,
        total_layers: int,
        layer: list[str],
    ) -> None:
        pass

    def _report_execution_complete(
        self,
        execution_id: str,
        num_outputs: int,
    ) -> None:
        self._schedule_broadcast(
            {
                "type": "execution_complete",
                "execution_id": execution_id,
            }
        )

    def _handle_execution_error(
        self,
        error: Exception,
        execution_id: str,
    ) -> dict[str, Any]:
        full_traceback = traceback.format_exc()
        logger.error(
            "Execution %s failed: %s: %s\n%s",
            execution_id,
            type(error).__name__,
            error,
            full_traceback,
        )

        error_info = {
            "type": type(error).__name__,
            "message": str(error),
        }

        self._schedule_broadcast(
            {
                "type": "execution_error",
                "execution_id": execution_id,
                "error": error_info,
            }
        )

        return {
            "execution_id": execution_id,
            "error": str(error),
        }

    async def execute(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import uuid

        execution_id = str(uuid.uuid4())
        self._current_execution_id = execution_id
        self._cancel_event.clear()

        logger.info(
            "[%s] === EXECUTION START === nodes=%d, edges=%d, global_timeout=%ds",
            execution_id[:8],
            len(nodes),
            len(edges),
            self.global_timeout,
        )

        self._report_execution_start(execution_id, len(nodes), len(edges))

        try:
            # Wrap entire execution in global timeout
            return await asyncio.wait_for(
                self._execute_workflow(nodes, edges, execution_id),
                timeout=self.global_timeout,
            )
        except TimeoutError:
            logger.error(
                "[%s] === GLOBAL TIMEOUT === execution exceeded %ds",
                execution_id[:8],
                self.global_timeout,
            )
            await self._broadcast(
                {
                    "type": "execution_error",
                    "execution_id": execution_id,
                    "error": {
                        "type": "GlobalTimeoutError",
                        "message": f"Execution timed out after {self.global_timeout}s",
                    },
                }
            )
            return {
                "execution_id": execution_id,
                "error": f"Execution timed out after {self.global_timeout}s",
            }
        except ExecutionCancelledError:
            logger.info("[%s] Execution cancelled", execution_id[:8])
            await self._broadcast(
                {
                    "type": "execution_cancelled",
                    "execution_id": execution_id,
                }
            )
            return {
                "execution_id": execution_id,
                "cancelled": True,
            }
        except Exception as e:
            logger.exception(
                "[%s] === EXECUTION FAILED === %s",
                execution_id[:8],
                e,
            )
            return self._handle_execution_error(e, execution_id)
        finally:
            self._current_execution_id = None
            self._cancel_event.clear()

    def _apply_feedback_inputs(
        self,
        node_id: str,
        inputs: dict[str, Any],
        feedback_connections: list[tuple[str, str, str, str]],
        node_outputs: dict[str, dict[str, Any]],
    ) -> None:
        for src, tgt, src_h, tgt_h in feedback_connections:
            if tgt == node_id and src in node_outputs:
                src_outputs = node_outputs[src]
                if src_h in src_outputs:
                    inputs[tgt_h] = src_outputs[src_h]

    def _apply_previous_state_inputs(
        self,
        node_id: str,
        inputs: dict[str, Any],
        node_outputs: dict[str, dict[str, Any]],
    ) -> None:
        if node_id not in node_outputs:
            return
        for key, value in node_outputs[node_id].items():
            if key.startswith("_"):
                inputs[f"_prev_{key.removeprefix('_')}"] = value

    async def _execute_layer(
        self,
        layer_index: int,
        layer: list[str],
        node_instances: dict[str, BaseNode],
        connections: Any,
        feedback_connections: list[tuple[str, str, str, str]],
        node_outputs: dict[str, dict[str, Any]],
        context: ExecutionContext,
        execution_id: str,
    ) -> list[tuple[str, Any]]:
        self._check_cancelled()

        logger.info(
            "[%s] Starting layer %d/%d with %d node(s): %s",
            execution_id[:8],
            layer_index,
            len(layer) - 1,
            len(layer),
            layer,
        )

        self._report_layer_start(execution_id, layer_index, len(layer), layer)

        tasks: list[tuple[str, Any]] = []
        for node_id in layer:
            node = node_instances[node_id]
            inputs = self._gather_node_inputs(
                node_id, connections, node_outputs
            )
            self._apply_feedback_inputs(
                node_id, inputs, feedback_connections, node_outputs
            )
            self._apply_previous_state_inputs(node_id, inputs, node_outputs)
            task = self._execute_single_node(
                node_id, node, inputs, context, execution_id
            )
            tasks.append((node_id, task))

        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (node_id, _), result in zip(tasks, results, strict=True):
            if isinstance(result, ExecutionCancelledError):
                raise result
            if isinstance(result, BaseException):
                raise result
            node_outputs[node_id] = result

        return tasks

    def _is_iteration_done(
        self,
        tasks: list[tuple[str, Any]],
        node_outputs: dict[str, dict[str, Any]],
        execution_id: str,
        iteration: int,
        max_iterations: int,
    ) -> bool:
        current_iteration_nodes = {node_id for node_id, _ in tasks}
        for node_id in current_iteration_nodes:
            if node_outputs.get(node_id, {}).get("done") is True:
                logger.info(
                    "[%s] Termination signal from node %s",
                    execution_id[:8],
                    node_id,
                )
                return True

        if iteration >= max_iterations:
            logger.warning(
                "[%s] Max iterations (%d) reached",
                execution_id[:8],
                max_iterations,
            )
            return True

        return False

    async def _run_iteration(
        self,
        iteration: int,
        execution_layers: list[list[str]],
        node_instances: dict[str, BaseNode],
        connections: Any,
        feedback_connections: list[tuple[str, str, str, str]],
        node_outputs: dict[str, dict[str, Any]],
        context: ExecutionContext,
        execution_id: str,
        has_feedback: bool,
    ) -> list[tuple[str, Any]]:
        context.round_num = iteration
        self._check_cancelled()

        if has_feedback:
            logger.info(
                "[%s] === ITERATION %d START ===",
                execution_id[:8],
                iteration,
            )
            await self._broadcast(
                {
                    "type": "iteration_start",
                    "execution_id": execution_id,
                    "iteration": iteration,
                }
            )

        last_tasks: list[tuple[str, Any]] = []
        for layer_index, layer in enumerate(execution_layers):
            last_tasks = await self._execute_layer(
                layer_index,
                layer,
                node_instances,
                connections,
                feedback_connections,
                node_outputs,
                context,
                execution_id,
            )

        return last_tasks

    async def _execute_workflow(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        execution_id: str,
    ) -> dict[str, Any]:
        (
            node_instances,
            _dependencies,
            connections,
            execution_layers,
            feedback_connections,
        ) = self._build_execution_graph(nodes, edges)

        has_feedback = len(feedback_connections) > 0
        max_iterations = 20

        logger.info(
            "[%s] Execution plan: %d layers, %d feedback edges",
            execution_id[:8],
            len(execution_layers),
            len(feedback_connections),
        )

        self._validate_workflow(node_instances)

        context, node_outputs = self._initialize_execution_state(execution_id)

        iteration = 0
        while True:
            iteration += 1
            last_tasks = await self._run_iteration(
                iteration,
                execution_layers,
                node_instances,
                connections,
                feedback_connections,
                node_outputs,
                context,
                execution_id,
                has_feedback,
            )

            if not has_feedback:
                break

            if self._is_iteration_done(
                last_tasks,
                node_outputs,
                execution_id,
                iteration,
                max_iterations,
            ):
                break

        context.node_outputs = node_outputs

        logger.info(
            "[%s] === EXECUTION COMPLETE === iterations=%d, outputs from %d nodes",
            execution_id[:8],
            iteration,
            len(node_outputs),
        )

        self._report_execution_complete(execution_id, len(node_outputs))

        return {
            "execution_id": execution_id,
            "outputs": node_outputs,
        }

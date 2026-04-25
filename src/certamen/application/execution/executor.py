# pylint: disable=duplicate-code
import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from certamen.application.workflow.nodes.base import BaseNode
from certamen.application.workflow.registry import registry
from certamen.domain.errors import GraphValidationError
from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


def _truncate_for_log(value: Any, max_length: int = 200) -> str:
    str_value = json.dumps(value, default=str, ensure_ascii=False)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    return str_value


DEFAULT_NODE_TIMEOUT = 300  # 5 minutes per node


class BaseExecutor(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.node_timeout = self.config.get(
            "node_timeout", DEFAULT_NODE_TIMEOUT
        )

    async def _execute_node_with_timeout(
        self,
        node: BaseNode,
        inputs: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        import asyncio

        try:
            return await asyncio.wait_for(
                node.execute(inputs, context),
                timeout=self.node_timeout,
            )
        except TimeoutError as err:
            raise TimeoutError(
                f"Node {node.NODE_TYPE} timed out after {self.node_timeout}s"
            ) from err

    def _build_adjacency_map(
        self,
        node_ids: set[str],
        edges: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for edge in edges:
            source = edge["source"]
            if source in adjacency:
                adjacency[source].append(edge["target"])
        return adjacency

    def _has_path_excluding_edge(
        self,
        adjacency: dict[str, list[str]],
        start: str,
        end: str,
        excluded_edge: dict[str, str],
    ) -> bool:
        from collections import deque

        visited: set[str] = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == end:
                return True
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adjacency.get(current, []):
                if (
                    excluded_edge["source"] == current
                    and excluded_edge["target"] == neighbor
                ):
                    continue
                if neighbor not in visited:
                    queue.append(neighbor)
        return False

    def _classify_edge(
        self,
        edge: dict[str, Any],
        adjacency: dict[str, list[str]],
    ) -> bool:
        source = edge["source"]
        target = edge["target"]
        is_feedback = self._has_path_excluding_edge(
            adjacency, target, source, edge
        )
        if is_feedback:
            logger.info(
                "Detected feedback edge: %s.%s -> %s.%s",
                source,
                edge.get("sourceHandle"),
                target,
                edge.get("targetHandle"),
            )
        return is_feedback

    def _detect_feedback_edges(
        self,
        node_ids: set[str],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        adjacency = self._build_adjacency_map(node_ids, edges)
        normal_edges = []
        feedback_edges = []
        for edge in edges:
            if self._classify_edge(edge, adjacency):
                feedback_edges.append(edge)
            else:
                normal_edges.append(edge)
        return normal_edges, feedback_edges

    def _build_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[
        dict[str, BaseNode],
        dict[str, list[str]],
        dict[str, list[tuple[str, str, str]]],
    ]:
        logger.debug(
            "Building graph: %d nodes, %d edges", len(nodes), len(edges)
        )

        node_instances: dict[str, BaseNode] = {}
        errors: list[str] = []

        for node_data in nodes:
            node_id = node_data["id"]
            node_type = node_data["type"]
            properties = node_data.get("properties", {})

            logger.debug(
                "Node [%s] type=%s properties=%s",
                node_id,
                node_type,
                _truncate_for_log(properties),
            )

            node_instance = registry.create(node_type, node_id, properties)
            if node_instance:
                node_instances[node_id] = node_instance
            else:
                errors.append(f"Unknown node type: {node_type}")

        if errors:
            raise GraphValidationError(
                f"Graph build errors: {'; '.join(errors)}"
            )

        dependencies: dict[str, list[str]] = defaultdict(list)
        connections: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            source_handle = edge.get("sourceHandle", "output")
            target_handle = edge.get("targetHandle", "input")

            logger.debug(
                "Edge: %s.%s -> %s.%s",
                source,
                source_handle,
                target,
                target_handle,
            )

            if source not in node_instances:
                raise GraphValidationError(
                    f"Edge references unknown source node: {source}"
                )
            if target not in node_instances:
                raise GraphValidationError(
                    f"Edge references unknown target node: {target}"
                )

            dependencies[target].append(source)
            connections[target].append((source, source_handle, target_handle))

        logger.debug(
            "Graph built successfully: %d nodes, %d dependencies",
            len(node_instances),
            sum(len(d) for d in dependencies.values()),
        )

        return node_instances, dict(dependencies), dict(connections)

    def _topological_sort(
        self,
        nodes: dict[str, BaseNode],
        dependencies: dict[str, list[str]],
    ) -> list[str]:
        from collections import deque

        in_degree: dict[str, int] = dict.fromkeys(nodes, 0)

        for node_id, deps in dependencies.items():
            if node_id in in_degree:
                in_degree[node_id] = len(deps)

        queue = deque(
            node_id for node_id, degree in in_degree.items() if degree == 0
        )
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            for other_id, deps in dependencies.items():
                if node_id in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(result) != len(nodes):
            raise GraphValidationError(
                "Graph contains a cycle - check node connections"
            )

        return result

    def _compute_in_degrees(
        self,
        nodes: dict[str, BaseNode],
        dependencies: dict[str, list[str]],
    ) -> dict[str, int]:
        in_degree: dict[str, int] = dict.fromkeys(nodes, 0)
        for node_id, deps in dependencies.items():
            if node_id in in_degree:
                in_degree[node_id] = len(deps)
        return in_degree

    def _reduce_in_degrees_for_layer(
        self,
        layer: list[str],
        dependencies: dict[str, list[str]],
        in_degree: dict[str, int],
        processed: set[str],
    ) -> None:
        for node_id in layer:
            for other_id, deps in dependencies.items():
                if node_id in deps and other_id not in processed:
                    in_degree[other_id] -= 1

    def _build_execution_layers(
        self,
        nodes: dict[str, BaseNode],
        dependencies: dict[str, list[str]],
    ) -> list[list[str]]:
        in_degree = self._compute_in_degrees(nodes, dependencies)
        layers: list[list[str]] = []
        processed: set[str] = set()

        while len(processed) < len(nodes):
            current_layer = [
                node_id
                for node_id in nodes
                if node_id not in processed and in_degree[node_id] == 0
            ]

            if not current_layer:
                raise GraphValidationError(
                    "Graph contains a cycle - check node connections"
                )

            layers.append(current_layer)
            processed.update(current_layer)
            self._reduce_in_degrees_for_layer(
                current_layer, dependencies, in_degree, processed
            )

        return layers

    def _validate_required_inputs(
        self,
        node: BaseNode,
        inputs: dict[str, Any],
    ) -> list[str]:
        missing = []
        for port in node.INPUTS:
            if port.required and port.name not in inputs:
                missing.append(port.name)
        return missing

    def _log_node_start(
        self,
        execution_id: str,
        node_id: str,
        node_type: str,
        inputs: dict[str, Any],
    ) -> None:
        logger.info(
            "[%s] Starting node [%s] (%s)",
            execution_id[:8],
            node_id,
            node_type,
        )
        logger.debug(
            "[%s] Node [%s] inputs: %s",
            execution_id[:8],
            node_id,
            _truncate_for_log(inputs, 500),
        )

    def _log_node_complete(
        self,
        execution_id: str,
        node_id: str,
        outputs: dict[str, Any],
    ) -> None:
        logger.info(
            "[%s] Node [%s] completed with %d outputs",
            execution_id[:8],
            node_id,
            len(outputs),
        )
        logger.debug(
            "[%s] Node [%s] outputs: %s",
            execution_id[:8],
            node_id,
            _truncate_for_log(outputs, 500),
        )

    def _log_missing_inputs(
        self,
        execution_id: str,
        node_id: str,
        missing_inputs: list[str],
    ) -> None:
        logger.warning(
            "[%s] Node [%s] missing required inputs: %s",
            execution_id[:8],
            node_id,
            missing_inputs,
        )

    def _log_node_error(
        self,
        execution_id: str,
        node_id: str,
        node_type: str,
        error: Exception,
    ) -> None:
        logger.exception(
            "[%s] Node [%s] (%s) execution failed: %s: %s",
            execution_id[:8],
            node_id,
            node_type,
            type(error).__name__,
            error,
        )

    def _validate_workflow(
        self,
        node_instances: dict[str, BaseNode],
    ) -> None:
        if not node_instances:
            raise GraphValidationError("No valid nodes in graph")

    def _build_execution_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[
        dict[str, BaseNode],
        dict[str, list[str]],
        dict[str, list[tuple[str, str, str]]],
        list[list[str]],
        list[
            tuple[str, str, str, str]
        ],  # feedback_connections: (source, target, source_handle, target_handle)
    ]:
        node_ids = {n["id"] for n in nodes}
        normal_edges, feedback_edges = self._detect_feedback_edges(
            node_ids, edges
        )

        node_instances, dependencies, connections = self._build_graph(
            nodes, normal_edges
        )

        feedback_connections: list[tuple[str, str, str, str]] = []
        for edge in feedback_edges:
            source = edge["source"]
            target = edge["target"]
            source_handle = edge.get("sourceHandle", "output")
            target_handle = edge.get("targetHandle", "input")
            feedback_connections.append(
                (source, target, source_handle, target_handle)
            )

        execution_layers = self._build_execution_layers(
            node_instances, dependencies
        )
        return (
            node_instances,
            dependencies,
            connections,
            execution_layers,
            feedback_connections,
        )

    def _gather_node_inputs(
        self,
        node_id: str,
        connections: dict[str, list[tuple[str, str, str]]],
        node_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        node_connections = connections.get(node_id, [])

        for source_id, source_handle, target_handle in node_connections:
            if source_id in node_outputs:
                source_outputs = node_outputs[source_id]
                if source_handle in source_outputs:
                    value = source_outputs[source_handle]
                    inputs[target_handle] = value
                    logger.debug(
                        "Input [%s.%s] <- [%s.%s]: %s",
                        node_id,
                        target_handle,
                        source_id,
                        source_handle,
                        _truncate_for_log(value),
                    )
                else:
                    logger.debug(
                        "Input [%s.%s]: source handle %s not found in %s outputs (available: %s)",
                        node_id,
                        target_handle,
                        source_handle,
                        source_id,
                        list(source_outputs.keys()),
                    )

        if inputs:
            logger.debug(
                "Node [%s] gathered %d inputs: %s",
                node_id,
                len(inputs),
                list(inputs.keys()),
            )
        else:
            logger.debug("Node [%s] has no inputs", node_id)

        return inputs

    def validate(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        errors = []
        warnings = []

        node_ids = {n["id"] for n in nodes}
        normal_edges, feedback_edges = self._detect_feedback_edges(
            node_ids, edges
        )

        if feedback_edges:
            feedback_info = [
                f"{e['source']} -> {e['target']}" for e in feedback_edges
            ]
            warnings.append(
                f"Detected {len(feedback_edges)} feedback edge(s): {feedback_info}"
            )

        try:
            node_instances, dependencies, connections = self._build_graph(
                nodes, normal_edges
            )
        except GraphValidationError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": warnings,
            }

        try:
            self._topological_sort(node_instances, dependencies)
        except GraphValidationError as e:
            errors.append(str(e))

        for node_id, node in node_instances.items():
            required_inputs = [p.name for p in node.INPUTS if p.required]
            connected_inputs = set()

            for _, _, target_handle in connections.get(node_id, []):
                connected_inputs.add(target_handle)

            missing = set(required_inputs) - connected_inputs
            if missing:
                warnings.append(
                    f"Node {node_id} ({node.DISPLAY_NAME}) missing required inputs: {list(missing)}"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    @abstractmethod
    def _initialize_execution_state(
        self,
        execution_id: str,
    ) -> tuple[Any, dict[str, dict[str, Any]]]: ...

    @abstractmethod
    async def _execute_single_node(
        self,
        node_id: str,
        node: BaseNode,
        inputs: dict[str, Any],
        context: Any,
        execution_id: str,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _report_execution_start(
        self,
        execution_id: str,
        num_nodes: int,
        num_edges: int,
    ) -> None: ...

    @abstractmethod
    def _report_layer_start(
        self,
        execution_id: str,
        layer_index: int,
        total_layers: int,
        layer: list[str],
    ) -> None: ...

    @abstractmethod
    def _report_execution_complete(
        self,
        execution_id: str,
        num_outputs: int,
    ) -> None: ...

    @abstractmethod
    def _handle_execution_error(
        self,
        error: Exception,
        execution_id: str,
    ) -> dict[str, Any]: ...

    def _log_workflow_inputs(
        self,
        execution_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        for node_data in nodes:
            logger.debug(
                "[%s] Workflow node: [%s] type=%s props=%s",
                execution_id[:8],
                node_data.get("id", "unknown"),
                node_data.get("type", "unknown"),
                _truncate_for_log(node_data.get("properties", {})),
            )
        for edge in edges:
            logger.debug(
                "[%s] Workflow edge: %s.%s -> %s.%s",
                execution_id[:8],
                edge.get("source"),
                edge.get("sourceHandle"),
                edge.get("target"),
                edge.get("targetHandle"),
            )

    def _log_execution_plan(
        self,
        execution_id: str,
        execution_layers: list[list[str]],
        feedback_connections: list[tuple[str, str, str, str]],
    ) -> None:
        logger.info(
            "[%s] Execution plan: %d layers, %d feedback edges",
            execution_id[:8],
            len(execution_layers),
            len(feedback_connections),
        )
        for i, layer in enumerate(execution_layers):
            logger.debug("[%s] Layer %d: %s", execution_id[:8], i, layer)

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

    def _inject_previous_state_inputs(
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

    async def _run_layer(
        self,
        execution_id: str,
        layer_index: int,
        layer: list[str],
        execution_layers: list[list[str]],
        node_instances: dict[str, BaseNode],
        connections: dict[str, list[tuple[str, str, str]]],
        feedback_connections: list[tuple[str, str, str, str]],
        node_outputs: dict[str, dict[str, Any]],
        context: Any,
    ) -> list[tuple[str, Any]]:
        logger.info(
            "[%s] Starting layer %d/%d with %d node(s): %s",
            execution_id[:8],
            layer_index,
            len(execution_layers) - 1,
            len(layer),
            layer,
        )
        self._report_layer_start(
            execution_id, layer_index, len(execution_layers), layer
        )

        tasks: list[tuple[str, Any]] = []
        for node_id in layer:
            node = node_instances[node_id]
            inputs = self._gather_node_inputs(
                node_id, connections, node_outputs
            )
            self._apply_feedback_inputs(
                node_id, inputs, feedback_connections, node_outputs
            )
            self._inject_previous_state_inputs(node_id, inputs, node_outputs)
            task = self._execute_single_node(
                node_id, node, inputs, context, execution_id
            )
            tasks.append((node_id, task))

        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (node_id, _), result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                raise result
            node_outputs[node_id] = result

        logger.info(
            "[%s] Completed layer %d/%d",
            execution_id[:8],
            layer_index,
            len(execution_layers) - 1,
        )
        return tasks

    def _check_termination(
        self,
        execution_id: str,
        tasks: list[tuple[str, Any]],
        node_outputs: dict[str, dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> bool:
        for node_id, _ in tasks:
            if (
                node_id in node_outputs
                and node_outputs[node_id].get("done") is True
            ):
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

    async def _run_execution_loop(
        self,
        execution_id: str,
        node_instances: dict[str, BaseNode],
        connections: dict[str, list[tuple[str, str, str]]],
        execution_layers: list[list[str]],
        feedback_connections: list[tuple[str, str, str, str]],
        context: Any,
        node_outputs: dict[str, dict[str, Any]],
    ) -> int:
        has_feedback = len(feedback_connections) > 0
        max_iterations = 20
        iteration = 0

        while True:
            iteration += 1
            context.round_num = iteration

            if has_feedback:
                logger.info(
                    "[%s] === ITERATION %d START ===",
                    execution_id[:8],
                    iteration,
                )

            last_tasks: list[tuple[str, Any]] = []
            for layer_index, layer in enumerate(execution_layers):
                last_tasks = await self._run_layer(
                    execution_id,
                    layer_index,
                    layer,
                    execution_layers,
                    node_instances,
                    connections,
                    feedback_connections,
                    node_outputs,
                    context,
                )

            if not has_feedback:
                break

            if self._check_termination(
                execution_id,
                last_tasks,
                node_outputs,
                iteration,
                max_iterations,
            ):
                break

        return iteration

    async def execute(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import uuid

        execution_id = str(uuid.uuid4())

        logger.info(
            "[%s] === EXECUTION START === nodes=%d, edges=%d",
            execution_id[:8],
            len(nodes),
            len(edges),
        )
        self._report_execution_start(execution_id, len(nodes), len(edges))
        self._log_workflow_inputs(execution_id, nodes, edges)

        try:
            (
                node_instances,
                _dependencies,
                connections,
                execution_layers,
                feedback_connections,
            ) = self._build_execution_graph(nodes, edges)

            self._log_execution_plan(
                execution_id, execution_layers, feedback_connections
            )
            self._validate_workflow(node_instances)

            context, node_outputs = self._initialize_execution_state(
                execution_id
            )

            iteration = await self._run_execution_loop(
                execution_id,
                node_instances,
                connections,
                execution_layers,
                feedback_connections,
                context,
                node_outputs,
            )

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

        except Exception as e:
            logger.exception(
                "[%s] === EXECUTION FAILED === %s",
                execution_id[:8],
                e,
            )
            return self._handle_execution_error(e, execution_id)

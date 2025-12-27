# pylint: disable=duplicate-code
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from arbitrium_core.application.workflow.nodes.base import BaseNode
from arbitrium_core.application.workflow.registry import registry
from arbitrium_core.domain.errors import GraphValidationError
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


def _truncate_for_log(value: Any, max_length: int = 200) -> str:
    """Truncate value for logging to avoid huge log entries."""
    str_value = json.dumps(value, default=str, ensure_ascii=False)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    return str_value


DEFAULT_NODE_TIMEOUT = 300  # 5 minutes per node


class BaseExecutor(ABC):
    """Abstract base class for workflow executors.

    Executors are responsible for:
    - Validating workflow graphs (nodes + edges)
    - Building execution graphs with dependency resolution
    - Executing nodes in topologically sorted order
    - Handling errors and timeouts
    - Returning execution results

    Implementations:
    - AsyncExecutor: async execution with event broadcasting
    - SyncExecutor: synchronous blocking execution with console output (CLI mode)
    """

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

    def _detect_feedback_edges(
        self,
        node_ids: set[str],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Detect edges that create cycles (feedback edges).

        Returns:
            tuple: (normal_edges, feedback_edges)
        """
        from collections import deque

        adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if source in adjacency:
                adjacency[source].append(target)

        def has_path(
            start: str, end: str, excluded_edge: dict[str, str]
        ) -> bool:
            """Check if path exists from start to end without using excluded_edge."""
            visited = set()
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

        normal_edges = []
        feedback_edges = []

        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if has_path(target, source, edge):
                feedback_edges.append(edge)
                logger.info(
                    "Detected feedback edge: %s.%s -> %s.%s",
                    source,
                    edge.get("sourceHandle"),
                    target,
                    edge.get("targetHandle"),
                )
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

    def _build_execution_layers(
        self,
        nodes: dict[str, BaseNode],
        dependencies: dict[str, list[str]],
    ) -> list[list[str]]:
        """
        Build execution layers for parallel execution.
        Returns list of layers, where each layer contains node IDs that can be executed in parallel.
        Example: [[node1, node2], [node3], [node4, node5]] means:
          - Layer 0: node1 and node2 run in parallel
          - Layer 1: node3 runs after layer 0 completes
          - Layer 2: node4 and node5 run in parallel after layer 1 completes
        """
        in_degree: dict[str, int] = dict.fromkeys(nodes, 0)

        for node_id, deps in dependencies.items():
            if node_id in in_degree:
                in_degree[node_id] = len(deps)

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

            for node_id in current_layer:
                for other_id, deps in dependencies.items():
                    if node_id in deps and other_id not in processed:
                        in_degree[other_id] -= 1

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
        except ValueError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": warnings,
            }

        try:
            self._topological_sort(node_instances, dependencies)
        except ValueError as e:
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
    ) -> tuple[Any, dict[str, dict[str, Any]]]:
        """Initialize execution context and outputs dict.

        Subclasses implement this to create appropriate ExecutionContext
        (with or without broadcast callback).
        """

    @abstractmethod
    async def _execute_single_node(
        self,
        node_id: str,
        node: BaseNode,
        inputs: dict[str, Any],
        context: Any,
        execution_id: str,
    ) -> dict[str, Any]:
        """Execute a single node with progress reporting.

        Subclasses implement this to add appropriate progress reporting
        (WebSocket broadcast or console output).
        """

    @abstractmethod
    def _report_execution_start(
        self,
        execution_id: str,
        num_nodes: int,
        num_edges: int,
    ) -> None:
        """Report execution start."""

    @abstractmethod
    def _report_layer_start(
        self,
        execution_id: str,
        layer_index: int,
        total_layers: int,
        layer: list[str],
    ) -> None:
        """Report layer execution start."""

    @abstractmethod
    def _report_execution_complete(
        self,
        execution_id: str,
        num_outputs: int,
    ) -> None:
        """Report successful execution completion."""

    @abstractmethod
    def _handle_execution_error(
        self,
        error: Exception,
        execution_id: str,
    ) -> dict[str, Any]:
        """Handle execution error and return error response."""

    async def execute(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute workflow graph.

        Args:
            nodes: List of node definitions
            edges: List of edge definitions

        Returns:
            dict with keys:
                - execution_id (str): Unique execution identifier
                - outputs (dict[str, dict[str, Any]]): Node outputs by node_id
                - error (str, optional): Error message if execution failed
        """
        import uuid

        execution_id = str(uuid.uuid4())

        logger.info(
            "[%s] === EXECUTION START === nodes=%d, edges=%d",
            execution_id[:8],
            len(nodes),
            len(edges),
        )

        self._report_execution_start(execution_id, len(nodes), len(edges))

        for node_data in nodes:
            node_type = node_data.get("type", "unknown")
            nid = node_data.get("id", "unknown")
            props = node_data.get("properties", {})
            logger.debug(
                "[%s] Workflow node: [%s] type=%s props=%s",
                execution_id[:8],
                nid,
                node_type,
                _truncate_for_log(props),
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

        try:
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
            for i, layer in enumerate(execution_layers):
                logger.debug("[%s] Layer %d: %s", execution_id[:8], i, layer)

            self._validate_workflow(node_instances)

            context, node_outputs = self._initialize_execution_state(
                execution_id
            )

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

                for layer_index, layer in enumerate(execution_layers):
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

                    tasks = []
                    for node_id in layer:
                        node = node_instances[node_id]
                        inputs = self._gather_node_inputs(
                            node_id, connections, node_outputs
                        )
                        for src, tgt, src_h, tgt_h in feedback_connections:
                            if tgt == node_id and src in node_outputs:
                                src_outputs = node_outputs[src]
                                if src_h in src_outputs:
                                    inputs[tgt_h] = src_outputs[src_h]
                        # Inject node's own previous outputs (for state persistence)
                        if node_id in node_outputs:
                            prev_outputs = node_outputs[node_id]
                            for key, value in prev_outputs.items():
                                if key.startswith("_"):
                                    inputs[f"_prev{key}"] = value
                        task = self._execute_single_node(
                            node_id, node, inputs, context, execution_id
                        )
                        tasks.append((node_id, task))

                    import asyncio

                    results = await asyncio.gather(
                        *[task for _, task in tasks], return_exceptions=True
                    )

                    for (node_id, _), result in zip(
                        tasks, results, strict=False
                    ):
                        if isinstance(result, BaseException):
                            raise result
                        node_outputs[node_id] = result

                    logger.info(
                        "[%s] Completed layer %d/%d",
                        execution_id[:8],
                        layer_index,
                        len(execution_layers) - 1,
                    )

                if not has_feedback:
                    break

                done = False
                for node_id, outputs in node_outputs.items():
                    if outputs.get("done") is True:
                        done = True
                        logger.info(
                            "[%s] Termination signal from node %s",
                            execution_id[:8],
                            node_id,
                        )
                        break

                if done:
                    break

                if iteration >= max_iterations:
                    logger.warning(
                        "[%s] Max iterations (%d) reached",
                        execution_id[:8],
                        max_iterations,
                    )
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

        except Exception as e:
            logger.exception(
                "[%s] === EXECUTION FAILED === %s",
                execution_id[:8],
                e,
            )
            return self._handle_execution_error(e, execution_id)

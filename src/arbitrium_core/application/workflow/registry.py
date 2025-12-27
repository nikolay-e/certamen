from typing import Any

from arbitrium_core.application.workflow.nodes.base import BaseNode


class NodeRegistry:
    _instance: "NodeRegistry | None" = None
    _nodes: dict[str, type[BaseNode]] = {}

    def __new__(cls) -> "NodeRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nodes = {}
        return cls._instance

    def register(self, node_class: type[BaseNode]) -> type[BaseNode]:
        self._nodes[node_class.NODE_TYPE] = node_class
        return node_class

    def get(self, node_type: str) -> type[BaseNode] | None:
        return self._nodes.get(node_type)

    def create(
        self,
        node_type: str,
        node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> BaseNode | None:
        node_class = self.get(node_type)
        if node_class:
            return node_class(node_id, properties)
        return None

    def list_nodes(self) -> list[dict[str, Any]]:
        return [
            node_class(node_id="schema").get_schema()
            for node_class in self._nodes.values()
            if not node_class.HIDDEN
        ]

    def list_by_category(self) -> dict[str, list[dict[str, Any]]]:
        by_category: dict[str, list[dict[str, Any]]] = {}
        for node_class in self._nodes.values():
            if node_class.HIDDEN:
                continue
            category = node_class.CATEGORY
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(
                node_class(node_id="schema").get_schema()
            )

        category_order = ["LLM", "Tournament"]
        sorted_result: dict[str, list[dict[str, Any]]] = {}
        for cat in category_order:
            if cat in by_category:
                sorted_result[cat] = by_category[cat]
        for cat in sorted(by_category.keys()):
            if cat not in sorted_result:
                sorted_result[cat] = by_category[cat]
        return sorted_result


registry = NodeRegistry()


def register_node(node_class: type[BaseNode]) -> type[BaseNode]:
    return registry.register(node_class)

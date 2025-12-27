from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class WorkflowSerializer(ABC):
    @abstractmethod
    def load_from_file(self, path: str | Path) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_from_string(self, content: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def to_executor_format(self, workflow: dict[str, Any]) -> dict[str, Any]:
        pass

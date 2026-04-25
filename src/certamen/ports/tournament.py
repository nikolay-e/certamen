from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class EventHandler(ABC):
    @abstractmethod
    def publish(self, _event_name: str, _data: dict[str, Any]) -> None:
        pass


class HostEnvironment(ABC):
    base_dir: Path | str

    @abstractmethod
    async def read_file(self, path: str) -> str:
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        pass

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        pass

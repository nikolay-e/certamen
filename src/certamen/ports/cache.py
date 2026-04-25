from typing import Protocol


class CacheProtocol(Protocol):
    def get(
        self, model_name: str, prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, float] | None: ...

    def set(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response: str,
        cost: float,
    ) -> None: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...

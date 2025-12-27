from typing import Any, Protocol


class SecretsProvider(Protocol):
    def get_secret(self, key: str) -> str | None: ...

    def load_secrets(
        self, config: dict[str, Any], required_providers: list[str]
    ) -> None: ...

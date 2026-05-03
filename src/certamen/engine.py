import asyncio
from pathlib import Path
from typing import Any

from certamen.infrastructure.config.loader import Config
from certamen.infrastructure.similarity import TfidfSimilarityEngine
from certamen.ports.llm import BaseModel, ModelResponse
from certamen.ports.similarity import SimilarityEngine
from certamen.ports.tournament import HostEnvironment
from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class _InternalHost(HostEnvironment):
    def __init__(self, base_dir: str | None):
        if base_dir is None:
            base_dir = "."
        self.base_dir = Path(base_dir).resolve()

    def _validate_path(self, path: str) -> Path:
        resolved = (Path(self.base_dir) / path).resolve()
        if not resolved.is_relative_to(self.base_dir):
            raise ValueError(
                f"Path traversal detected: '{path}' escapes base directory"
            )
        return resolved

    async def write_file(self, path: str, content: str) -> None:
        file_path = self._validate_path(path)
        await asyncio.to_thread(
            file_path.parent.mkdir, parents=True, exist_ok=True
        )
        await asyncio.to_thread(
            file_path.write_text, content, encoding="utf-8"
        )

    async def read_file(self, path: str) -> str:
        file_path = self._validate_path(path)
        return await asyncio.to_thread(file_path.read_text, encoding="utf-8")

    def get_secret(self, key: str) -> str | None:
        import os

        return os.getenv(key)


class Certamen:
    def __init__(
        self,
        config: Config,
        all_models: dict[str, BaseModel],
        healthy_models: dict[str, BaseModel],
        failed_models: dict[str, Exception],
        similarity_engine: SimilarityEngine | None = None,
    ) -> None:
        outputs_dir = config.config_data.get("outputs_dir")

        self.config = config
        self._all_models = all_models
        self._healthy_models: dict[str, Any] = healthy_models
        self._failed_models = failed_models
        self._similarity_engine = similarity_engine or TfidfSimilarityEngine()

        self._host = _InternalHost(
            base_dir=str(outputs_dir) if outputs_dir else None
        )
        self._outputs_dir = (
            Path(outputs_dir) if outputs_dir else Path("reports")
        )

    @classmethod
    async def from_settings(
        cls,
        settings: dict[str, Any],
        skip_secrets: bool = False,
        skip_health_check: bool = False,
    ) -> "Certamen":
        from certamen.application.bootstrap import build_certamen

        return await build_certamen(
            settings,
            skip_secrets=skip_secrets,
            skip_health_check=skip_health_check,
        )

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
        skip_secrets: bool = False,
        skip_health_check: bool = False,
    ) -> "Certamen":
        from certamen.application.bootstrap import build_certamen

        return await build_certamen(
            config_path,
            skip_secrets=skip_secrets,
            skip_health_check=skip_health_check,
        )

    @property
    def healthy_models(self) -> dict[str, BaseModel]:
        return self._healthy_models

    @property
    def all_models(self) -> dict[str, BaseModel]:
        return self._all_models

    @property
    def failed_models(self) -> dict[str, Exception]:
        return self._failed_models

    @property
    def config_data(self) -> dict[str, Any]:
        return self.config.config_data

    @property
    def is_ready(self) -> bool:
        return len(self._healthy_models) > 0

    @property
    def healthy_model_count(self) -> int:
        return len(self._healthy_models)

    @property
    def failed_model_count(self) -> int:
        return len(self._failed_models)

    async def run_single_model(
        self, model_key: str, prompt: str
    ) -> ModelResponse:
        if model_key not in self._healthy_models:
            if model_key in self._failed_models:
                raise ValueError(
                    f"Model '{model_key}' failed health check: "
                    f"{self._failed_models[model_key]}"
                )
            raise KeyError(f"Model '{model_key}' not found in configuration")

        model: BaseModel = self._healthy_models[model_key]
        return await model.generate(prompt)

    async def run_all_models(self, prompt: str) -> dict[str, ModelResponse]:
        if not self._healthy_models:
            raise RuntimeError("No healthy models available")

        logger.info(
            "Running prompt through %d models", len(self._healthy_models)
        )

        model_keys = list(self._healthy_models.keys())

        async def _run_one(key: str) -> tuple[str, ModelResponse | None]:
            try:
                return key, await self.run_single_model(key, prompt)
            except Exception:
                logger.exception("Failed to run %s", key)
                return key, None

        pairs = await asyncio.gather(*[_run_one(k) for k in model_keys])
        return {k: r for k, r in pairs if r is not None}

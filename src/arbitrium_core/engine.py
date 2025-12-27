import asyncio
from pathlib import Path
from typing import Any

from arbitrium_core.domain.tournament.tournament import ModelComparison
from arbitrium_core.infrastructure.config.loader import Config
from arbitrium_core.infrastructure.similarity import TfidfSimilarityEngine
from arbitrium_core.ports.llm import BaseModel, ModelResponse
from arbitrium_core.ports.similarity import SimilarityEngine
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class _InternalEventHandler:
    def publish(self, _event_name: str, _data: dict[str, Any]) -> None:
        pass


class _InternalHost:
    def __init__(self, base_dir: str | None):
        if base_dir is None:
            base_dir = "."
        self.base_dir = Path(base_dir).resolve()

    def _validate_path(self, path: str) -> Path:
        """Validate path is within base_dir to prevent path traversal attacks."""
        resolved = (self.base_dir / path).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
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


class Arbitrium:
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
        self._healthy_models = healthy_models
        self._failed_models = failed_models
        self._last_comparison: ModelComparison | None = None
        self._similarity_engine = similarity_engine or TfidfSimilarityEngine()

        self._event_handler = _InternalEventHandler()
        self._host = _InternalHost(
            base_dir=str(outputs_dir) if outputs_dir else None
        )

    @classmethod
    async def from_settings(
        cls,
        settings: dict[str, Any],
        skip_secrets: bool = False,
        skip_health_check: bool = False,
    ) -> "Arbitrium":
        from arbitrium_core.application.bootstrap import build_arbitrium

        return await build_arbitrium(
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
    ) -> "Arbitrium":
        from arbitrium_core.application.bootstrap import build_arbitrium

        return await build_arbitrium(
            config_path,
            skip_secrets=skip_secrets,
            skip_health_check=skip_health_check,
        )

    # === Public Properties ===

    @property
    def healthy_models(self) -> dict[str, BaseModel]:
        """Get only models that passed health check."""
        return self._healthy_models

    @property
    def all_models(self) -> dict[str, BaseModel]:
        """Get all models (including those that failed health check)."""
        return self._all_models

    @property
    def failed_models(self) -> dict[str, Exception]:
        """Get models that failed health check with their errors."""
        return self._failed_models

    @property
    def config_data(self) -> dict[str, Any]:
        """Get configuration data dictionary."""
        return self.config.config_data

    @property
    def is_ready(self) -> bool:
        """Whether the framework has healthy models ready to use."""
        return len(self._healthy_models) > 0

    @property
    def healthy_model_count(self) -> int:
        """Number of healthy models available."""
        return len(self._healthy_models)

    @property
    def failed_model_count(self) -> int:
        """Number of models that failed health check."""
        return len(self._failed_models)

    # === Execution Methods ===

    async def run_tournament(
        self,
        question: str,
        models: dict[str, BaseModel] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if models is None:
            models = self._healthy_models

        if not models:
            raise RuntimeError(
                "No healthy models available to run tournament. "
                f"Failed models: {list(self._failed_models.keys())}"
            )

        logger.info("Starting tournament with %d models", len(models))

        comparison = self._create_comparison(models)
        self._last_comparison = comparison
        result = await comparison.run(question)

        logger.info("Tournament completed successfully")

        # Build metrics dictionary
        metrics = {
            "total_cost": comparison.total_cost,
            "champion_model": (
                comparison.active_model_keys[0]
                if comparison.active_model_keys
                else None
            ),
            "active_model_keys": comparison.active_model_keys.copy(),
            "eliminated_models": comparison.eliminated_models.copy(),
            "cost_by_model": comparison.cost_by_model.copy(),
        }

        return result, metrics

    async def run_single_model(
        self, model_key: str, prompt: str
    ) -> ModelResponse:
        if model_key not in self._healthy_models:
            if model_key in self._failed_models:
                raise ValueError(
                    f"Model '{model_key}' failed health check: {self._failed_models[model_key]}"
                )
            raise KeyError(f"Model '{model_key}' not found in configuration")

        model = self._healthy_models[model_key]
        return await model.generate(prompt)

    async def run_all_models(self, prompt: str) -> dict[str, ModelResponse]:
        if not self._healthy_models:
            raise RuntimeError("No healthy models available")

        logger.info(
            "Running prompt through %d models", len(self._healthy_models)
        )

        results = {}
        for model_key in self._healthy_models:
            try:
                response = await self.run_single_model(model_key, prompt)
                results[model_key] = response
            except Exception:
                logger.exception("Failed to run %s", model_key)
                # Continue with other models
                continue

        return results

    # === Internal Methods ===

    def _create_comparison(
        self,
        models: dict[str, BaseModel] | None = None,
    ) -> ModelComparison:
        if models is None:
            models = self._healthy_models

        return ModelComparison(
            config=self.config_data,
            models=models,
            event_handler=self._event_handler,  # type: ignore[arg-type]
            host=self._host,  # type: ignore[arg-type]
            similarity_engine=self._similarity_engine,
        )

import asyncio
from pathlib import Path
from typing import Any

from certamen.domain.tournament.tournament import ModelComparison
from certamen.infrastructure.config.loader import Config
from certamen.infrastructure.events import JsonlEventHandler, generate_run_id
from certamen.infrastructure.similarity import TfidfSimilarityEngine
from certamen.ports.llm import BaseModel, ModelResponse
from certamen.ports.similarity import SimilarityEngine
from certamen.ports.tournament import EventHandler, HostEnvironment
from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class _NoopEventHandler(EventHandler):
    def publish(self, _event_name: str, _data: dict[str, Any]) -> None:
        pass


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
        self._comparison: ModelComparison | None = None
        self._failed_models = failed_models
        self._last_comparison: ModelComparison | None = None
        self._similarity_engine = similarity_engine or TfidfSimilarityEngine()

        self._event_handler: EventHandler = _NoopEventHandler()
        self._host = _InternalHost(
            base_dir=str(outputs_dir) if outputs_dir else None
        )
        self._outputs_dir = Path(outputs_dir) if outputs_dir else Path(".")
        self.last_run_id: str | None = None
        self.last_run_dir: Path | None = None

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

    # === Public Properties ===

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

        run_id = generate_run_id()
        run_dir = self._outputs_dir / "runs" / run_id
        self.last_run_id = run_id
        self.last_run_dir = run_dir

        previous_handler = self._event_handler
        with JsonlEventHandler(run_dir, run_id) as jsonl_handler:
            self._event_handler = jsonl_handler
            jsonl_handler.publish(
                "tournament_started",
                {
                    "run_id": run_id,
                    "question": question,
                    "models": list(models.keys()),
                    "config_features": self.config_data.get("features", {}),
                },
            )
            try:
                comparison = self._create_comparison(models)
                self._last_comparison = comparison
                result = await comparison.run(question)
                jsonl_handler.publish(
                    "tournament_ended",
                    {
                        "champion": (
                            comparison.active_model_keys[0]
                            if comparison.active_model_keys
                            else None
                        ),
                        "total_cost": comparison.total_cost,
                        "eliminated": [
                            e.get("model")
                            for e in comparison.eliminated_models
                        ],
                    },
                )
            finally:
                self._event_handler = previous_handler

        logger.info("Tournament completed successfully (run_id=%s)", run_id)

        # Build metrics dictionary
        metrics: dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "total_cost": comparison.total_cost,
            "champion_model": (
                comparison.active_model_keys[0]
                if comparison.active_model_keys
                else None
            ),
            "active_model_keys": comparison.active_model_keys.copy(),
            "eliminated_models": comparison.eliminated_models.copy(),
            "cost_by_model": comparison.cost_by_model.copy(),
            "knowledge_map": getattr(
                comparison.runner, "_knowledge_map", None
            ),
        }

        return result, metrics

    async def run_deep_extraction(
        self,
        question: str,
        depth: int = 2,
    ) -> tuple[str, Any]:
        if not self.config_data.get("features", {}).get(
            "deep_extraction_enabled", False
        ):
            return await self.run_tournament(question)

        result, metrics = await self.run_tournament(question)
        km = metrics.get("knowledge_map")

        if km is None or depth <= 1:
            return result, metrics

        branches = km.exploration_branches[:3]
        for i, branch in enumerate(branches):
            try:
                logger.info(
                    "Deep extraction branch %d/%d (depth %d): %s",
                    i + 1,
                    len(branches),
                    depth,
                    branch[:80],
                )
                _sub_result, sub_metrics = await self.run_deep_extraction(
                    branch, depth=depth - 1
                )
                sub_km = sub_metrics.get("knowledge_map")
                if sub_km is None:
                    continue
                km.consensus.extend(sub_km.consensus)
                km.disagreements.extend(sub_km.disagreements)
                for sub_model, sub_insights in sub_km.unique_insights.items():
                    km.unique_insights.setdefault(sub_model, []).extend(
                        sub_insights
                    )
                km.known_unknowns.extend(sub_km.known_unknowns)
            except Exception as e:
                logger.warning(
                    "Deep extraction branch '%s' failed: %s", branch[:80], e
                )

        metrics["knowledge_map"] = km
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

        model: BaseModel = self._healthy_models[model_key]
        return await model.generate(prompt)

    async def run_all_models(self, prompt: str) -> dict[str, ModelResponse]:
        import asyncio

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
            event_handler=self._event_handler,
            host=self._host,
            similarity_engine=self._similarity_engine,
        )

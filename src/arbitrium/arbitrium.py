"""
Arbitrium - Unified framework interface for LLM tournament evaluation.

This module provides the single entry point for the Arbitrium Framework.
All initialization, execution, and configuration is handled through the Arbitrium class.
"""

from pathlib import Path
from typing import Any

from arbitrium.config.loader import Config
from arbitrium.core.comparison import ModelComparison
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import LiteLLMModel, ModelResponse
from arbitrium.models.factory import create_models_from_config
from arbitrium.utils.constants import HEALTH_CHECK_PROMPT
from arbitrium.utils.exceptions import ConfigurationError
from arbitrium.utils.secrets import load_secrets

logger = get_contextual_logger("arbitrium")


class _InternalEventHandler:
    """Internal event handler for framework operations."""

    def publish(self, _event_name: str, _data: dict[str, Any]) -> None:
        """Publish events - no-op for now, could be extended for callbacks."""
        pass


class _InternalHost:
    """Internal host for file operations."""

    def __init__(self, base_dir: str):
        """
        Initialize host with output directory.

        Args:
            base_dir: Output directory path (REQUIRED - no defaults)

        Raises:
            ValueError: If base_dir is not provided
        """
        if not base_dir:
            raise ValueError("outputs_dir is required by framework - no default values allowed")
        self.base_dir = Path(base_dir)

    async def write_file(self, path: str, content: str) -> None:
        """Write file to disk."""
        file_path = self.base_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        """Read file from disk."""
        file_path = self.base_dir / path
        return file_path.read_text(encoding="utf-8")

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment."""
        import os

        return os.getenv(key)


class Arbitrium:
    """
    Arbitrium Framework - LLM tournament evaluation system.

    This is the main entry point for the framework. It handles:
    - Configuration loading
    - Model initialization and health checking
    - Tournament execution
    - Single model queries

    Example:
        >>> # Initialize from config file
        >>> arbitrium = await Arbitrium.from_config("config.yml")
        >>>
        >>> # Run tournament
        >>> result = await arbitrium.run_tournament("Your question here")
        >>>
        >>> # Or run single model
        >>> response = await arbitrium.run_single_model("gpt-4", "Hello!")
    """

    def __init__(
        self,
        config: Config,
        all_models: dict[str, LiteLLMModel],
        healthy_models: dict[str, LiteLLMModel],
        failed_models: dict[str, Exception],
    ) -> None:
        """
        Initialize Arbitrium with pre-loaded components.

        Args:
            config: Loaded configuration object (must contain outputs_dir)
            all_models: All models from config (including failed ones)
            healthy_models: Models that passed health check
            failed_models: Models that failed health check with their errors

        Raises:
            ValueError: If outputs_dir is not in config
        """
        # Get outputs_dir from config - framework validates this
        outputs_dir = config.config_data.get("outputs_dir")
        if not outputs_dir:
            raise ValueError(
                "outputs_dir is required in configuration file. "
                "Framework does not provide default values. "
                "Add 'outputs_dir: <path>' to your config file."
            )

        self.config = config
        self._all_models = all_models
        self._healthy_models = healthy_models
        self._failed_models = failed_models

        # Internal components - not exposed to users
        self._event_handler = _InternalEventHandler()
        self._host = _InternalHost(base_dir=str(outputs_dir))

    @classmethod
    async def from_settings(
        cls,
        settings: dict[str, Any],
        skip_secrets: bool = False,
        skip_health_check: bool = False,
    ) -> "Arbitrium":
        """
        Create Arbitrium from a settings dictionary.

        This is the most flexible entry point - allows creating the framework
        without a config file. Useful for web applications, testing, or
        programmatic usage.

        Args:
            settings: Configuration dictionary (must contain 'models', 'retry', 'outputs_dir', etc.)
            skip_secrets: If True, don't load secrets from environment/1Password
            skip_health_check: If True, skip model health checking

        Returns:
            Initialized Arbitrium instance with healthy models filtered

        Raises:
            ConfigurationError: If settings dictionary is invalid or missing outputs_dir

        Example:
            >>> settings = {
            >>>     "models": {"gpt-4": {...}},
            >>>     "retry": {...},
            >>>     "features": {...},
            >>>     "prompts": {...},
            >>>     "outputs_dir": "./results"
            >>> }
            >>> arbitrium = await Arbitrium.from_settings(settings)
        """
        from arbitrium.config.loader import validate_config

        # Step 1: Validate settings (including outputs_dir)
        logger.info("Validating settings dictionary")
        is_valid, errors = validate_config(settings)
        if not is_valid:
            error_details = "\n  - ".join(errors)
            raise ConfigurationError(
                f"Invalid configuration provided. Missing or invalid sections:\n  - {error_details}\n\n"
                f"Required sections: models, retry, features, prompts, outputs_dir"
            )

        # Create a Config object and populate it
        config_obj = Config()
        config_obj.config_data = settings
        config_obj._setup_config_shortcuts()

        # Step 2: Load secrets
        if not skip_secrets:
            cls._load_secrets(settings)

        # Step 3: Create models
        logger.info("Creating models from settings")
        all_models = create_models_from_config(settings)

        if not all_models:
            logger.warning("No models configured in settings")
            return cls(
                config=config_obj,
                all_models={},
                healthy_models={},
                failed_models={},
            )

        # Step 4: Health check (optional)
        healthy_models = dict(all_models)  # Start with all models
        failed_models: dict[str, Exception] = {}

        if not skip_health_check:
            healthy_models, failed_models = await cls._health_check_models(all_models)

        logger.info(f"Arbitrium initialized: {len(healthy_models)} healthy, {len(failed_models)} failed")

        return cls(
            config=config_obj,
            all_models=all_models,
            healthy_models=healthy_models,
            failed_models=failed_models,
        )

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
        skip_secrets: bool = False,
        skip_health_check: bool = False,
    ) -> "Arbitrium":
        """
        Create Arbitrium from configuration file.

        This is the most common way to initialize the framework.

        Args:
            config_path: Path to YAML configuration file (must contain outputs_dir field)
            skip_secrets: If True, don't load secrets from environment/1Password
            skip_health_check: If True, skip model health checking

        Returns:
            Initialized Arbitrium instance with healthy models filtered

        Raises:
            ConfigurationError: If config file cannot be loaded, is invalid, or missing outputs_dir

        Example:
            >>> arbitrium = await Arbitrium.from_config("config.yml")
            >>> result = await arbitrium.run_tournament("What is the meaning of life?")
        """
        # Load configuration from file
        config_path = Path(config_path)
        logger.info(f"Loading configuration from {config_path}")
        config_obj = Config(str(config_path))

        if not config_obj.load():
            raise ConfigurationError(f"Failed to load configuration from {config_path}")

        # Delegate to from_settings (outputs_dir will be validated there)
        return await cls.from_settings(
            settings=config_obj.config_data,
            skip_secrets=skip_secrets,
            skip_health_check=skip_health_check,
        )

    @staticmethod
    def _load_secrets(config: dict[str, Any]) -> None:
        """
        Load secrets for configured models.

        Args:
            config: Configuration dictionary
        """
        # Check if all providers are local (don't require secrets)
        active_providers = {model_cfg.get("provider", "").lower() for model_cfg in config.get("models", {}).values() if model_cfg.get("provider")}

        local_providers = {"ollama"}
        if active_providers and active_providers.issubset(local_providers):
            logger.info("All models use local providers, skipping secret loading")
            return

        # Skip if no secrets section in config
        if "secrets" not in config:
            logger.info("No secrets section in config, skipping secret loading")
            return

        try:
            load_secrets(config, list(active_providers))
        except ConfigurationError as e:
            logger.warning(f"Failed to load secrets: {e}")
            logger.warning("Continuing without secrets - remote models may fail")

    @staticmethod
    async def _health_check_models(
        models: dict[str, LiteLLMModel],
    ) -> tuple[dict[str, LiteLLMModel], dict[str, Exception]]:
        """
        Perform health check on all models.

        Args:
            models: Dictionary of models to check

        Returns:
            Tuple of (healthy_models, failed_models)
        """
        logger.info(f"Performing health check on {len(models)} models...")

        healthy: dict[str, LiteLLMModel] = {}
        failed: dict[str, Exception] = {}

        for model_key, model in models.items():
            try:
                response = await model.generate(HEALTH_CHECK_PROMPT)

                if response.is_error():
                    error = Exception(response.error or "Unknown error")
                    logger.warning(f"❌ {model.full_display_name}: {error}")
                    failed[model_key] = error
                else:
                    logger.info(f"✅ {model.full_display_name}: Healthy")
                    healthy[model_key] = model

            except Exception as e:
                logger.warning(f"❌ {model.full_display_name}: {e}")
                failed[model_key] = e

        logger.info(f"Health check complete: {len(healthy)} healthy, {len(failed)} failed")
        return healthy, failed

    # === Public Properties ===

    @property
    def healthy_models(self) -> dict[str, LiteLLMModel]:
        """Get only models that passed health check."""
        return self._healthy_models

    @property
    def all_models(self) -> dict[str, LiteLLMModel]:
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
        models: dict[str, LiteLLMModel] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Run a full Arbitrium tournament for the given question.

        This executes the complete tournament flow:
        1. Create ModelComparison
        2. Run tournament with all healthy models
        3. Return the champion's final response and metrics

        Args:
            question: The question to analyze
            models: Optional models dict (defaults to healthy_models)

        Returns:
            Tuple of (champion_response, metrics_dict)

            metrics_dict contains:
            - total_cost: Total API cost for tournament
            - champion_model: Key of winning model
            - active_model_keys: List of remaining models
            - eliminated_models: List of eliminated model keys
            - cost_by_model: Dict of model -> cost

        Raises:
            RuntimeError: If no healthy models are available

        Example:
            >>> result, metrics = await arbitrium.run_tournament(
            >>>     "Should we use microservices or monolith?"
            >>> )
            >>> print(f"Winner: {metrics['champion_model']}")
            >>> print(f"Cost: ${metrics['total_cost']:.4f}")
        """
        if models is None:
            models = self._healthy_models

        if not models:
            raise RuntimeError("No healthy models available to run tournament. " f"Failed models: {list(self._failed_models.keys())}")

        logger.info(f"Starting tournament with {len(models)} models")

        comparison = self._create_comparison(models)
        result = await comparison.run(question)

        logger.info("Tournament completed successfully")

        # Build metrics dictionary
        metrics = {
            "total_cost": comparison.total_cost,
            "champion_model": comparison.active_model_keys[0] if comparison.active_model_keys else None,
            "active_model_keys": comparison.active_model_keys.copy(),
            "eliminated_models": comparison.eliminated_models.copy(),
            "cost_by_model": comparison.cost_by_model.copy(),
        }

        return result, metrics

    async def run_single_model(self, model_key: str, prompt: str) -> ModelResponse:
        """
        Run a single model with the given prompt.

        This is a simpler alternative to tournaments when you just want
        one model's response.

        Args:
            model_key: Key of the model to use (must be in healthy_models)
            prompt: The prompt to send to the model

        Returns:
            ModelResponse from the model

        Raises:
            KeyError: If model_key not found in healthy models
            ValueError: If model failed health check

        Example:
            >>> response = await arbitrium.run_single_model("gpt-4", "Hello!")
            >>> print(response.content)
        """
        if model_key not in self._healthy_models:
            if model_key in self._failed_models:
                raise ValueError(f"Model '{model_key}' failed health check: {self._failed_models[model_key]}")
            raise KeyError(f"Model '{model_key}' not found in configuration")

        model = self._healthy_models[model_key]
        return await model.generate(prompt)

    async def run_all_models(self, prompt: str) -> dict[str, ModelResponse]:
        """
        Run the same prompt through all healthy models.

        Useful for comparisons or benchmarking.

        Args:
            prompt: The prompt to send to all models

        Returns:
            Dictionary mapping model keys to their responses

        Example:
            >>> responses = await arbitrium.run_all_models("Explain quantum computing")
            >>> for model_key, response in responses.items():
            >>>     print(f"{model_key}: {response.content[:100]}...")
        """
        if not self._healthy_models:
            raise RuntimeError("No healthy models available")

        logger.info(f"Running prompt through {len(self._healthy_models)} models")

        results = {}
        for model_key in self._healthy_models:
            try:
                response = await self.run_single_model(model_key, prompt)
                results[model_key] = response
            except Exception as e:
                logger.error(f"Failed to run {model_key}: {e}")
                # Continue with other models
                continue

        return results

    # === Internal Methods ===

    def _create_comparison(
        self,
        models: dict[str, LiteLLMModel] | None = None,
    ) -> ModelComparison:
        """
        Create a ModelComparison instance for tournament execution.

        Args:
            models: Optional models dict (defaults to healthy_models)

        Returns:
            Initialized ModelComparison instance
        """
        if models is None:
            models = self._healthy_models

        return ModelComparison(
            config=self.config_data,
            models=models,  # type: ignore[arg-type]
            event_handler=self._event_handler,  # type: ignore[arg-type]
            host=self._host,  # type: ignore[arg-type]
        )

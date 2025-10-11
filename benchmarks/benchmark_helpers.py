"""
Utility functions for benchmarks to avoid code duplication.

This module provides backward-compatible wrappers around Arbitrium
for existing benchmark code.
"""

import asyncio

from arbitrium import Arbitrium
from arbitrium.core.comparison import ModelComparison
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import LiteLLMModel

logger = get_contextual_logger("arbitrium.utils.benchmark")


def initialize_benchmark(
    config_path: str, skip_secrets: bool = False, skip_health_check: bool = False
) -> tuple[dict[str, object], dict[str, LiteLLMModel], ModelComparison]:
    """
    Initialize benchmark components (config, models, comparison).

    This is a backward-compatible wrapper around Arbitrium.
    New code should use Arbitrium.from_config() directly.

    Args:
        config_path: Path to configuration file
        skip_secrets: If True, skip loading secrets from environment/1Password
        skip_health_check: If True, skip model health checking

    Returns:
        Tuple of (config, models dict, ModelComparison instance)
    """
    logger.info("Initializing benchmark (using Arbitrium)")

    # Initialize Arbitrium (handles config, secrets, models, health check)
    arbitrium = asyncio.run(
        Arbitrium.from_config(
            config_path=config_path,
            skip_secrets=skip_secrets,
            skip_health_check=skip_health_check,
        )
    )

    # Create comparison with healthy models
    comparison = arbitrium._create_comparison()

    # Return in old format for backward compatibility
    return arbitrium.config_data, arbitrium.healthy_models, comparison

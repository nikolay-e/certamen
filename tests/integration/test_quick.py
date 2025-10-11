#!/usr/bin/env python3
"""Quick integration test for Arbitrium Framework tournament."""

import argparse
import asyncio
import sys

from arbitrium import Arbitrium
from arbitrium.logging import get_contextual_logger, setup_logging


async def main() -> None:
    """Run a quick integration test of the tournament system."""
    parser = argparse.ArgumentParser(description="Quick integration test of Arbitrium Framework tournament")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.example.yml",
        help="Path to config file (default: config.example.yml)",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What is the meaning of life?",
        help="Question for the tournament (default: 'What is the meaning of life?')",
    )
    args = parser.parse_args()

    # Setup logging (console only, no log files)
    setup_logging(verbose=args.verbose)
    logger = get_contextual_logger("test_quick")

    logger.info(f"Loading config from: {args.config}")

    # Initialize Arbitrium
    arbitrium = await Arbitrium.from_config(config_path=args.config)

    if not arbitrium.is_ready:
        logger.error("No healthy models available")
        sys.exit(1)

    logger.info("Models initialized:")
    for key, model in arbitrium.healthy_models.items():
        logger.info(f"  - {key}: {model.display_name}")

    question = args.question
    logger.info(f"Question: {question}")

    logger.info("Starting tournament...")
    result, metrics = await arbitrium.run_tournament(question)

    print(f"\n{'=' * 80}")
    print("TOURNAMENT RESULT:")
    print("=" * 80)
    print(result)
    print(f"\nChampion: {metrics['champion_model']}")
    print(f"Cost: ${metrics['total_cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

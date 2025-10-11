#!/usr/bin/env python3
"""
Micro-Benchmark: Single Model vs Arbitrium Framework

Purpose: Get first empirical data point showing Arbitrium Framework value.
Method: Run 1 high-stakes question through single model + CoT vs Arbitrium Framework.

This is NOT rigorous scientific validation - it's a quick proof point.

Configuration:
    - Question MUST be specified in config file under 'question:' field
    - No hardcoded questions - all test cases come from config
    - See config.micro_benchmark.yml for example structure

Usage:
    python -m arbitrium.benchmarks.micro_benchmark --config <path-to-config.yml>

Example:
    python -m arbitrium.benchmarks.micro_benchmark --config src/arbitrium/benchmarks/config.micro_benchmark.yml

"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

# Public API imports - benchmarks use only exported interface
from arbitrium import Arbitrium
from arbitrium.logging import get_contextual_logger, setup_logging

# Benchmark-specific imports
from benchmarks.reporting import generate_manual_evaluation_template

# Setup logging explicitly
setup_logging(verbose=True, enable_file_logging=True)
logger = get_contextual_logger("benchmarks.micro_benchmark")

# Chain-of-thought prompt for single model
COT_PROMPT = """
Think step-by-step. Consider multiple perspectives:
1. Technical architecture considerations
2. Team capacity and skill requirements
3. Business constraints and timeline
4. Risk factors and failure modes
5. Alternative approaches

Identify potential flaws in your reasoning. Then provide your recommendation.

Question: {question}
""".strip()


async def run_single_model_with_cot(question: str, model_name: str, arbitrium: Arbitrium) -> dict[str, Any]:
    """
    Run a single model with chain-of-thought prompting.

    Args:
        question: The question to ask
        model_name: Key of the model to use
        arbitrium: Arbitrium instance with loaded models

    Returns:
        Dictionary with results
    """
    model = arbitrium.healthy_models[model_name]

    logger.info(f"🤖 Running single model ({model.full_display_name}) with CoT prompting...")

    # Format CoT prompt
    prompt = COT_PROMPT.format(question=question)

    # Get response
    start_time = datetime.now()
    response_obj = await arbitrium.run_single_model(model_name, prompt)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    logger.info(f"✅ Single model response complete in {duration:.1f}s, cost: ${response_obj.cost:.4f}")

    return {
        "approach": "Single Model + Chain-of-Thought",
        "model": model.full_display_name,
        "response": response_obj.content,
        "duration_seconds": duration,
        "cost_estimate": response_obj.cost,
    }


async def run_arbitrium_tournament(question: str, arbitrium: Arbitrium) -> dict[str, Any]:
    """
    Run full Arbitrium Framework tournament.

    Args:
        question: The question to analyze
        arbitrium: Arbitrium instance

    Returns:
        Dictionary with tournament results
    """
    logger.info("🏆 Running Arbitrium Framework tournament...")

    # Run tournament using public API
    start_time = datetime.now()
    result, metrics = await arbitrium.run_tournament(question)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    logger.info(f"✅ Tournament complete in {duration:.1f}s, total cost: ${metrics['total_cost']:.4f}")

    return {
        "approach": "Arbitrium Framework Tournament",
        "champion_model": metrics["champion_model"] or "Unknown",
        "response": result,
        "duration_seconds": duration,
        "cost_actual": metrics["total_cost"],
        "cost_by_model": metrics.get("cost_by_model", {}),
        "eliminated_models": metrics["eliminated_models"],
    }


async def main(args: dict[str, Any] | None = None) -> None:
    """Run micro-benchmark."""
    if args is None:
        parser = argparse.ArgumentParser(description="Micro-Benchmark: Single Model vs Arbitrium Framework")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to configuration file (REQUIRED - no defaults)",
        )
        parsed_args = parser.parse_args()
        args = vars(parsed_args)  # Convert to dict

    config_path = args.get("config")
    if not config_path:
        raise ValueError("Configuration path is required. " "Specify with --config <path>. " "The framework does not use fallback configurations.")

    # Load config and inject outputs_dir
    from arbitrium.config.loader import Config

    config_obj = Config(config_path)
    if not config_obj.load():
        raise RuntimeError(f"Failed to load configuration from {config_path}")

    # Benchmark explicitly sets outputs_dir to current directory
    config_obj.config_data["outputs_dir"] = "."

    # Initialize Arbitrium (loads config, secrets, models, runs health check)
    logger.info("Initializing Arbitrium...")
    arbitrium = await Arbitrium.from_settings(settings=config_obj.config_data)

    # Get question from config
    question = arbitrium.config.config_data.get("question")
    if not question:
        raise ValueError(f"Configuration file {config_path} must contain a 'question' field. " "The benchmark does not use hardcoded questions.")

    print("=" * 80)
    print("MICRO-BENCHMARK: Single Model vs Arbitrium Framework")
    print("=" * 80)
    print()
    print("This benchmark provides a first data point for Arbitrium Framework value.")
    print("NOT scientifically rigorous - use for documentation and examples.")
    print()
    print(f"Config: {config_path}")
    print(f"\nTest Question:\n{question}")
    print()

    # Check if we have any healthy models
    if not arbitrium.is_ready:
        print("\n❌ ERROR: No healthy models available")
        if arbitrium.failed_models:
            print("\nFailed models:")
            for model_key, error in arbitrium.failed_models.items():
                print(f"  - {model_key}: {error}")
        return

    print(f"\n✅ {arbitrium.healthy_model_count} healthy models ready")
    if arbitrium.failed_model_count > 0:
        print(f"⚠️  {arbitrium.failed_model_count} models failed health check:")
        for model_key, error in arbitrium.failed_models.items():
            print(f"  - {model_key}: {error}")

    single_model_results = []

    # Run single models
    for model_name in arbitrium.healthy_models:
        print("\n" + "=" * 80)
        print(f"APPROACH 1: Single Model ({model_name}) + Chain-of-Thought")
        print("=" * 80 + "\n")

        try:
            single_result = await run_single_model_with_cot(question, model_name, arbitrium)
            single_model_results.append(single_result)

            print("\n✅ Complete!")
            print(f"   Duration: {single_result['duration_seconds']:.1f}s")
            print(f"   Cost: ${single_result['cost_estimate']:.4f}")
            if single_result["cost_estimate"] == 0.0:
                print("   (Local/free model - no API costs)")
        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            print(f"\n❌ Failed: {e}")
            print("   Continuing with other models...")

    # Run tournament
    print("\n" + "=" * 80)
    print("APPROACH 2: Arbitrium Framework Tournament")
    print("=" * 80 + "\n")

    tournament_result = await run_arbitrium_tournament(question, arbitrium)

    print("\n✅ Complete!")
    print(f"   Duration: {tournament_result['duration_seconds']:.1f}s ({tournament_result['duration_seconds'] / 60:.1f} min)")
    print(f"   Total Cost: ${tournament_result['cost_actual']:.4f}")

    # Show cost breakdown
    if tournament_result.get("cost_by_model"):
        print("\n   Cost Breakdown:")
        for model_key, cost in tournament_result["cost_by_model"].items():
            print(f"     - {model_key}: ${cost:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Print cost summary
    if single_model_results:
        print("\n📊 Cost Summary:")
        print("-" * 80)
        for i, result in enumerate(single_model_results):
            print(f"  Single Model #{i + 1} ({result['model']}): ${result['cost_estimate']:.4f}")
        print(f"  Arbitrium Tournament: ${tournament_result['cost_actual']:.4f}")

        # Show comparison
        avg_single_cost = sum(r["cost_estimate"] for r in single_model_results) / len(single_model_results)
        if avg_single_cost > 0:
            print(f"\n  Tournament vs Avg Single Model: {tournament_result['cost_actual'] / avg_single_cost:.2f}x cost")
        print("-" * 80)

    # Save reports relative to project root
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "micro_benchmark_results.md"

    with open(output_path, "w") as f:
        f.write("# Micro-Benchmark Results: Single Model vs Arbitrium Framework\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Config: `{config_path}`\n\n")

        f.write("## Test Question\n\n")
        f.write(f"{question}\n\n")

        f.write("---\n\n")

        for i, single_result in enumerate(single_model_results):
            f.write(f"## Approach 1.{i + 1}: Single Model\n\n")
            f.write(f"Model: {single_result['model']}\n\n")
            f.write(f"Duration: {single_result['duration_seconds']:.1f} seconds\n\n")
            f.write(f"Cost: ${single_result['cost_estimate']:.4f}\n\n")
            f.write("### Response\n\n")
            f.write(f"{single_result['response']}\n\n")

        f.write("---\n\n")

        f.write("## Approach 2: Arbitrium Framework Tournament\n\n")
        f.write(f"Champion: {tournament_result['champion_model']}\n\n")
        f.write(f"Duration: {tournament_result['duration_seconds']:.1f} seconds ({tournament_result['duration_seconds'] / 60:.1f} minutes)\n\n")
        f.write(f"Total Cost: ${tournament_result['cost_actual']:.4f}\n\n")

        # Cost breakdown by model
        if tournament_result.get("cost_by_model"):
            f.write("### Cost Breakdown by Model\n\n")
            for model_key, cost in tournament_result["cost_by_model"].items():
                f.write(f"- {model_key}: ${cost:.4f}\n")
            f.write("\n")

        # Cost and time comparisons
        if single_model_results:
            cost_estimate = single_model_results[0]["cost_estimate"]
            cost_actual = tournament_result["cost_actual"]
            if cost_estimate > 0:
                cost_multiple = f"{cost_actual / cost_estimate:.1f}x"
            else:
                cost_multiple = "N/A (free/local baseline)"
            f.write(f"Cost Multiple vs Single Model: {cost_multiple}\n\n")

            f.write(f"Time Multiple vs Single Model: {tournament_result['duration_seconds'] / single_model_results[0]['duration_seconds']:.1f}x\n\n")

        if tournament_result["eliminated_models"]:
            f.write(f"Eliminated Models: {', '.join(str(m) for m in tournament_result['eliminated_models'])}\n\n")

        f.write("### Champion Response\n\n")
        f.write(f"{tournament_result['response']}\n\n")

        f.write("---\n\n")

        model_names = [res["model"] for res in single_model_results] + ["Arbitrium Framework"]
        f.write(generate_manual_evaluation_template(model_names))

    print(f"\n📄 Results saved to: {output_path}")
    print()
    print("NEXT STEPS:")
    print("1. Share both responses with 3-5 knowledgeable people (anonymized)")
    print("2. Ask them to evaluate using the rubric")
    print("3. Compile feedback and calculate average scores")
    print("4. Document in README as first case study")
    print()
    print("This provides your first empirical evidence for Arbitrium Framework value!")


if __name__ == "__main__":
    asyncio.run(main())

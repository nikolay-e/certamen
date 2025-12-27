#!/usr/bin/env python3
"""
Arbitrium Framework - Quickstart Example

The absolute minimum code to run a tournament.
Perfect for: First-time users, quick validation.

Expected runtime: ~2 minutes
Expected cost: ~$0.50 (with 3 models)
"""

import asyncio

from arbitrium_core import Arbitrium


async def main():
    """Run a simple tournament on a strategic question."""

    # Initialize Arbitrium from your config file
    arbitrium = await Arbitrium.from_config("config.example.yml")

    print(f"✅ Loaded {arbitrium.healthy_model_count} healthy models")

    # Run tournament
    question = (
        "What is the best strategy for launching an open-source AI framework?"
    )

    print(f"\n🚀 Running tournament on: {question}\n")

    result, metrics = await arbitrium.run_tournament(question)

    # Display results
    print("\n" + "=" * 80)
    print("🏆 TOURNAMENT RESULTS")
    print("=" * 80)
    print(f"\n📊 Champion Model: {metrics['champion_model']}")
    print(f"💰 Total Cost: ${metrics['total_cost']:.4f}")
    print(f"🗑️  Eliminated: {len(metrics['eliminated_models'])} models")
    print("\n📝 Final Answer:\n")
    print(result[:500] + "..." if len(result) > 500 else result)

    # Cost breakdown
    print("\n💸 Cost Breakdown:")
    for model, cost in metrics["cost_by_model"].items():
        print(f"  - {model}: ${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

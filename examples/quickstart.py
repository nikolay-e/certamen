#!/usr/bin/env python3
"""
Certamen Framework - Quickstart Example

The absolute minimum code to run a tournament.
Perfect for: First-time users, quick validation.

Expected runtime: ~2 minutes
Expected cost: ~$0.50 (with 3 models)
"""

import asyncio

from certamen import Certamen


async def main():
    """Run a simple tournament on a strategic question."""

    # Initialize Certamen from your config file
    certamen = await Certamen.from_config("config.example.yml")

    print(f"✅ Loaded {certamen.healthy_model_count} healthy models")

    # Run tournament
    question = (
        "What is the best strategy for launching an open-source AI framework?"
    )

    print(f"\n🚀 Running tournament on: {question}\n")

    result, metrics = await certamen.run_tournament(question)

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

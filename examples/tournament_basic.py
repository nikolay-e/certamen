#!/usr/bin/env python3
"""
Certamen Framework - Basic Tournament Example

Run a tournament with default settings.
Perfect for: High-stakes decisions, synthesis of perspectives.

Expected runtime: ~5-10 minutes (depends on model count)
Expected cost: ~$0.50-2.00 (scales with models and rounds)
"""

import asyncio

from certamen import Certamen


async def main():
    """Run a basic tournament with detailed output."""

    # Initialize Certamen
    certamen = await Certamen.from_config("config.example.yml")

    # Verify models are ready
    print("=" * 80)
    print("🎯 CERTAMEN TOURNAMENT")
    print("=" * 80)
    print(f"\n✅ Healthy models: {certamen.healthy_model_count}")
    print(f"❌ Failed models: {certamen.failed_model_count}")

    if certamen.failed_model_count > 0:
        print("\n⚠️  Failed models:")
        for model_key, error in certamen.failed_models.items():
            print(f"   - {model_key}: {error}")

    print("\n🏃 Active participants:")
    for model_key in certamen.healthy_models.keys():
        print(f"   - {model_key}")

    # Define your strategic question
    question = """
    Analyze the strategic decision: Should our SaaS startup focus on
    product-led growth (PLG) or sales-led growth (SLG) for our first year?

    Context:
    - Pre-revenue, team of 3
    - B2B product (developer tools)
    - $500k seed funding
    - 12-month runway
    """

    print(f"\n❓ Question:\n{question}")
    print("\n🚀 Starting tournament...\n")

    # Run tournament
    result, metrics = await certamen.run_tournament(question)

    # Display comprehensive results
    print("\n" + "=" * 80)
    print("🏆 TOURNAMENT COMPLETE")
    print("=" * 80)

    print(f"\n🥇 Champion: {metrics['champion_model']}")
    print(f"💰 Total Cost: ${metrics['total_cost']:.4f}")
    print(f"🗑️  Eliminated Models: {len(metrics['eliminated_models'])}")

    if metrics["eliminated_models"]:
        print("\n   Elimination order:")
        for model in metrics["eliminated_models"]:
            print(f"   → {model}")

    print("\n💸 Cost Breakdown:")
    total = 0
    for model, cost in sorted(
        metrics["cost_by_model"].items(), key=lambda x: x[1], reverse=True
    ):
        print(
            f"   {model:20} ${cost:7.4f} ({cost/metrics['total_cost']*100:.1f}%)"
        )
        total += cost
    print(f"   {'TOTAL':20} ${total:7.4f}")

    print("\n" + "=" * 80)
    print("📝 CHAMPION SOLUTION")
    print("=" * 80)
    print(f"\n{result}\n")


if __name__ == "__main__":
    asyncio.run(main())

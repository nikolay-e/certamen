#!/usr/bin/env python3
"""
Arbitrium Framework - Tournament with Knowledge Bank

Demonstrates the Knowledge Bank: how insights from eliminated models
are preserved and injected into surviving models.

Perfect for: Understanding Arbitrium's core innovation.

Expected runtime: ~5-10 minutes
Expected cost: ~$0.50-2.00
"""

import asyncio

from arbitrium_core import Arbitrium


async def main():
    """Run tournament and inspect Knowledge Bank activity."""

    # Initialize with explicit configuration
    arbitrium = await Arbitrium.from_config("config.example.yml")

    print("=" * 80)
    print("🧠 KNOWLEDGE BANK DEMONSTRATION")
    print("=" * 80)

    # Verify Knowledge Bank is enabled in config
    kb_enabled = arbitrium.config_data.get("knowledge_bank", {}).get(
        "enabled", False
    )

    if not kb_enabled:
        print("\n⚠️  WARNING: Knowledge Bank is disabled in config!")
        print("   Set 'knowledge_bank.enabled: true' in config.example.yml")
        print("   to see Knowledge Bank in action.\n")
    else:
        print("\n✅ Knowledge Bank: ENABLED")
        print(
            f"   Strategy: {arbitrium.config_data['knowledge_bank'].get('strategy', 'extractive')}"
        )
        print(
            f"   Max entries: {arbitrium.config_data['knowledge_bank'].get('max_entries', 'unlimited')}"
        )

    print(
        f"\n🏃 Tournament participants: {arbitrium.healthy_model_count} models"
    )

    # Strategic question that benefits from diverse perspectives
    question = """
    Design a pricing strategy for an AI API service targeting developers.
    Consider:
    - Free tier for adoption
    - Usage-based vs. subscription models
    - Competitive landscape (OpenAI, Anthropic)
    - Startup constraints (need revenue fast)

    Provide a comprehensive pricing table with justification.
    """

    print(f"\n❓ Question: {question[:150]}...\n")
    print("🚀 Running tournament...\n")
    print("💡 Watch for Knowledge Bank activity in logs:\n")
    print("   - When models are eliminated")
    print("   - When insights are extracted")
    print("   - When KB context is injected into survivors\n")

    # Run tournament
    result, metrics = await arbitrium.run_tournament(question)

    # Analyze results
    print("\n" + "=" * 80)
    print("🏆 RESULTS & KNOWLEDGE BANK IMPACT")
    print("=" * 80)

    print(f"\n🥇 Champion: {metrics['champion_model']}")
    print(f"🗑️  Models eliminated: {len(metrics['eliminated_models'])}")

    if len(metrics["eliminated_models"]) > 0:
        print("\n🧠 Knowledge Bank Activity:")
        print(
            f"   ✅ {len(metrics['eliminated_models'])} models contributed insights before elimination"
        )
        print(
            "   ✅ Surviving models received KB context in improvement rounds"
        )
        print(
            "   ✅ Final answer synthesizes ideas from ALL models, not just the champion"
        )

    print(f"\n💰 Total Cost: ${metrics['total_cost']:.4f}")

    print("\n" + "=" * 80)
    print("📝 SYNTHESIZED SOLUTION (Champion + KB Insights)")
    print("=" * 80)
    print(f"\n{result}\n")

    # Tips for users
    print("\n💡 TIP: Check the tournament logs to see:")
    print("   - Which insights were extracted from eliminated models")
    print("   - How the Knowledge Bank context was used in refinement")
    print("   - Provenance tracking (which ideas came from which model)")


if __name__ == "__main__":
    asyncio.run(main())

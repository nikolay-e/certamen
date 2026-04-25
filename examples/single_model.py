#!/usr/bin/env python3
"""
Certamen Framework - Single Model Example

Use Certamen to query a single model without tournament.
Perfect for: Simple queries, cost-sensitive scenarios, rapid iteration.

Expected runtime: <30 seconds
Expected cost: ~$0.05-0.20 (per model)
"""

import asyncio

from certamen import Certamen


async def main():
    """Run a single model query."""

    # Initialize Certamen
    certamen = await Certamen.from_config("config.example.yml")

    print(f"✅ Available models: {list(certamen.healthy_models.keys())}\n")

    # Query a single model
    model_key = "gpt"  # Change to your model key from config
    question = "What are the top 3 risks in launching an AI startup?"

    print(f"🤖 Querying {model_key}...")
    print(f"❓ Question: {question}\n")

    response = await certamen.run_single_model(model_key, question)

    # Display results
    if response.is_error():
        print(f"❌ Error: {response.error}")
    else:
        print("📝 Response:")
        print("-" * 80)
        print(response.content)
        print("-" * 80)
        print(f"\n💰 Cost: ${response.cost:.4f}")
        print(f"📊 Provider: {response.provider}")


async def compare_models():
    """Compare responses from all models (no tournament)."""

    certamen = await Certamen.from_config("config.example.yml")

    question = "Should I use microservices or monolith architecture?"

    print("\n🔄 Running same question through all models...")
    print(f"❓ {question}\n")

    responses = await certamen.run_all_models(question)

    # Display comparison
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)

    for model_key, response in responses.items():
        if response.is_error():
            print(f"\n❌ {model_key}: Error - {response.error}")
        else:
            preview = (
                response.content[:200] + "..."
                if len(response.content) > 200
                else response.content
            )
            print(f"\n🤖 {model_key} (${response.cost:.4f}):")
            print(f"   {preview}")


if __name__ == "__main__":
    # Run single model
    asyncio.run(main())

    # Uncomment to compare all models
    # asyncio.run(compare_models())

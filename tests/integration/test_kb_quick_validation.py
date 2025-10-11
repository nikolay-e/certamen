#!/usr/bin/env python3
"""
Quick Knowledge Bank Validation Test

Purpose: Determine if KB adds measurable value with minimal testing.
Method: Run 3 test questions with KB ON vs KB OFF, manually inspect results.

Decision Criteria:
- If KB consistently adds novel, relevant insights → KEEP
- If KB shows marginal or inconsistent value → MAKE OPTIONAL
- If KB adds no value or noise → REMOVE

Usage:
    python tests/validation/test_kb_quick_validation.py
"""

import asyncio
from pathlib import Path
from typing import Any

from arbitrium.core.comparison import ModelComparison
from arbitrium.logging import get_contextual_logger
from benchmarks.benchmark_helpers import initialize_benchmark
from benchmarks.reporting import generate_manual_evaluation_template
from tests.test_data_shared import TEST_QUESTIONS

logger = get_contextual_logger("tests.integration.test_kb_quick_validation")


async def run_tournament(question: str, kb_enabled: bool, comparison: ModelComparison) -> dict[str, Any]:
    """Run a single tournament with KB enabled or disabled."""
    # Override KB setting
    comparison.config["knowledge_bank"]["enabled"] = kb_enabled

    # Run tournament
    result = await comparison.run(question)

    eliminated_models = getattr(comparison, "eliminated_models", [])
    return {
        "question": question,
        "kb_enabled": kb_enabled,
        "champion_answer": result,
        "eliminated_models": eliminated_models,
        "kb_insights": comparison.knowledge_bank.get_top_insights(num_insights=10) if kb_enabled else None,
    }


async def main() -> None:
    """Run quick validation test."""
    print("=" * 80)
    print("KNOWLEDGE BANK QUICK VALIDATION TEST")
    print("=" * 80)
    print()
    print("Running 3 test questions with KB ON and KB OFF...")
    print("This will take ~30-45 minutes total.")
    print()

    results = []

    # Initialize benchmark components once
    _config, _models, comparison = initialize_benchmark("config.yml")

    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(TEST_QUESTIONS)}: {test_case['domain']}")
        print(f"{'=' * 80}\n")

        # Run with KB OFF
        print("⏳ Running tournament with KB DISABLED...")
        result_kb_off = await run_tournament(test_case["question"], kb_enabled=False, comparison=comparison)

        print("\n" + "=" * 40)
        print("✅ KB OFF tournament complete")
        print("=" * 40 + "\n")

        # Run with KB ON
        print("⏳ Running tournament with KB ENABLED...")
        result_kb_on = await run_tournament(test_case["question"], kb_enabled=True, comparison=comparison)

        print("\n" + "=" * 40)
        print("✅ KB ON tournament complete")
        print("=" * 40 + "\n")

        # Store results
        results.append(
            {
                "domain": test_case["domain"],
                "question": test_case["question"],
                "kb_off": result_kb_off,
                "kb_on": result_kb_on,
            }
        )

        # Print KB insights for manual inspection
        kb_insights = result_kb_on["kb_insights"]
        if kb_insights:
            insights_list = list(kb_insights) if hasattr(kb_insights, "__iter__") else []
            print(f"\n📦 KB INSIGHTS EXTRACTED ({len(insights_list)} total):")
            print("-" * 80)
            for idx, insight in enumerate(insights_list[:5], 1):
                print(f"{idx}. {str(insight)[:150]}...")
            if len(insights_list) > 5:
                print(f"   ... and {len(insights_list) - 5} more")
            print("-" * 80)

    # Save results for manual review
    print("\n" + "=" * 80)
    print("VALIDATION TEST COMPLETE")
    print("=" * 80)
    print()
    print("Results saved to: tests/validation/kb_validation_results.txt")
    print()
    print("NEXT STEPS:")
    print("1. Manually review the champion answers (KB ON vs KB OFF)")
    print("2. Check KB insights - are they novel, relevant, actionable?")
    print("3. Make decision based on criteria:")
    print("   - KEEP: KB consistently adds value")
    print("   - MAKE OPTIONAL: Marginal benefit, let users decide\n")
    print("   - REMOVE: No benefit or adds noise")
    print()

    # Write results to file
    output_path = Path(__file__).parent / "kb_validation_results.txt"
    with open(output_path, "w") as f:
        f.write("KNOWLEDGE BANK VALIDATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"TEST CASE {i}: {result['domain']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question: {result['question']}\n\n")

            f.write("KB OFF - Champion Answer:\n")
            f.write(str(result["kb_off"]["champion_answer"]) + "\n\n")  # type: ignore[index]

            f.write("KB ON - Champion Answer:\n")
            f.write(str(result["kb_on"]["champion_answer"]) + "\n\n")  # type: ignore[index]

            kb_insights = result["kb_on"]["kb_insights"]  # type: ignore[index]
            if kb_insights:
                insights_list_file: list[Any] = list(kb_insights) if hasattr(kb_insights, "__iter__") else []
                f.write(f"KB Insights Extracted ({len(insights_list_file)}):\n")
                for idx, insight in enumerate(insights_list_file, 1):
                    f.write(f"{idx}. {insight!s}\n")
                f.write("\n")

            f.write("=" * 80 + "\n\n")

        f.write("\nMANUAL EVALUATION RUBRIC:\n")
        f.write("-" * 80 + "\n")
        f.write(generate_manual_evaluation_template(["KB ON", "KB OFF"]))
        f.write("DECISION:\n")
        f.write("Based on manual evaluation:\n")
        f.write("[ ] KEEP KB - Consistently adds significant value\n")
        f.write("[ ] MAKE OPTIONAL - Marginal benefit, let users decide\n")
        f.write("[ ] REMOVE KB - No benefit or degraded quality\n")


if __name__ == "__main__":
    asyncio.run(main())

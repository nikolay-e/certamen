#!/usr/bin/env python3
"""
Ablation Benchmark to validate the core hypotheses of Arbitrium Framework.

This benchmark compares four conditions:
A. Single model responses (baseline)
B. Single model with Chain-of-Thought prompting
C. Homogeneous tournament (one model family with different roles)
D. Heterogeneous tournament (standard Arbitrium with diverse models)

The goal is to demonstrate that:
1. Competition improves quality (C/D > A)
2. Diversity matters (D > C)
3. Tournaments outperform prompting techniques (D > B)

Usage:
    python -m benchmarks.ablation_benchmark --config config.example.yml --question "Your strategic question"

Example:
    python -m benchmarks.ablation_benchmark --config config.example.yml
"""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from arbitrium import Arbitrium
from arbitrium.logging import get_contextual_logger, setup_logging

# Setup logging
setup_logging(verbose=True, enable_file_logging=True)
logger = get_contextual_logger("ablation_benchmark")

# --- Constants and Configuration ---
OUTPUT_DIR = Path(__file__).parent / "reports" / "ablation"
HOMOGENEOUS_CONFIG_PATHS = {
    "gpt": "benchmarks/config.ablation.gpt_only.yml",
    "claude": "benchmarks/config.ablation.claude_only.yml",
    "gemini": "benchmarks/config.ablation.gemini_only.yml",
}

# Default test question (high-stakes strategic decision)
DEFAULT_QUESTION = """
Analyze the strategic decision: Should our SaaS startup focus on
product-led growth (PLG) or sales-led growth (SLG) for our first year?

Context:
- Pre-revenue, team of 3 engineers
- B2B product (developer tools for API testing)
- $500k seed funding, 12-month runway
- Competitive landscape: established players (Postman, Insomnia)
- Target: 10,000 users or $500k ARR by end of year

Consider:
- Customer acquisition costs
- Time to first revenue
- Scalability with small team
- Market expectations in this category
- Risk vs. speed trade-offs

Provide a clear recommendation with supporting evidence.
""".strip()

COT_PROMPT_TEMPLATE = """
Think step-by-step about this problem. Break down your analysis into clear stages:
1. Identify the key factors and constraints
2. Evaluate each option systematically
3. Consider potential risks and trade-offs
4. Synthesize your analysis into a recommendation

Question:
{question}
"""


# --- Runner Functions for Each Condition ---


async def run_condition_A(arbitrium: Arbitrium, question: str) -> dict[str, Any]:
    """
    Runs Condition A: Single model responses (baseline).

    Each model generates an independent response without any special prompting.
    This establishes the baseline quality.
    """
    logger.info("=" * 80)
    logger.info("CONDITION A: Single Model Responses (Baseline)")
    logger.info("=" * 80)

    results = {}
    costs = {}

    for model_key in arbitrium.healthy_models:
        model = arbitrium.healthy_models[model_key]
        logger.info(f"Running {model.full_display_name}...")

        response = await arbitrium.run_single_model(model_key, question)
        results[model_key] = {
            "content": response.content,
            "cost": response.cost,
            "model_name": model.model_name,
            "display_name": model.full_display_name,
        }
        costs[model_key] = response.cost

        logger.info(f"  Cost: ${response.cost:.4f}")

    total_cost = sum(costs.values())
    logger.info(f"\nCondition A total cost: ${total_cost:.4f}")

    return {
        "condition": "A",
        "description": "Single model responses (baseline)",
        "results": results,
        "total_cost": total_cost,
        "model_count": len(results),
    }


async def run_condition_B(arbitrium: Arbitrium, question: str, best_model_key: str) -> dict[str, Any]:
    """
    Runs Condition B: Best single model with Chain-of-Thought prompting.

    Uses the best-performing model from condition A with an explicit
    step-by-step reasoning prompt.
    """
    logger.info("=" * 80)
    logger.info(f"CONDITION B: Best Model ({best_model_key}) + Chain-of-Thought")
    logger.info("=" * 80)

    cot_prompt = COT_PROMPT_TEMPLATE.format(question=question)
    model = arbitrium.healthy_models[best_model_key]

    logger.info(f"Running {model.full_display_name} with CoT prompting...")
    response = await arbitrium.run_single_model(best_model_key, cot_prompt)

    logger.info(f"  Cost: ${response.cost:.4f}")

    return {
        "condition": "B",
        "description": f"Best model ({best_model_key}) with Chain-of-Thought",
        "model_key": best_model_key,
        "model_name": model.model_name,
        "display_name": model.full_display_name,
        "result": {
            "content": response.content,
            "cost": response.cost,
        },
        "total_cost": response.cost,
        "prompt_used": "chain_of_thought",
    }


async def run_condition_C(model_family: str, question: str) -> dict[str, Any]:
    """
    Runs Condition C: Homogeneous tournament.

    Same model family, different roles/perspectives via system prompts.
    Tests if role-playing improves quality even without model diversity.
    """
    logger.info("=" * 80)
    logger.info(f"CONDITION C: Homogeneous Tournament ({model_family.upper()})")
    logger.info("=" * 80)

    config_path = HOMOGENEOUS_CONFIG_PATHS[model_family]
    logger.info(f"Loading config: {config_path}")

    arbitrium_homo = await Arbitrium.from_config(config_path, skip_health_check=False)

    logger.info(f"Running tournament with {arbitrium_homo.healthy_model_count} instances...")
    result, metrics = await arbitrium_homo.run_tournament(question)

    logger.info(f"  Champion: {metrics['champion_model']}")
    logger.info(f"  Total cost: ${metrics['total_cost']:.4f}")

    return {
        "condition": "C",
        "description": f"Homogeneous tournament ({model_family})",
        "model_family": model_family,
        "champion_model": metrics["champion_model"],
        "champion_response": result,
        "metrics": metrics,
        "total_cost": metrics["total_cost"],
        "eliminated_models": metrics.get("eliminated_models", []),
    }


async def run_condition_D(arbitrium: Arbitrium, question: str) -> dict[str, Any]:
    """
    Runs Condition D: Heterogeneous tournament (standard Arbitrium).

    Multiple different model families competing. This is the full
    Arbitrium Framework as intended.
    """
    logger.info("=" * 80)
    logger.info("CONDITION D: Heterogeneous Tournament (Standard Arbitrium)")
    logger.info("=" * 80)

    logger.info(f"Running tournament with {arbitrium.healthy_model_count} diverse models...")
    result, metrics = await arbitrium.run_tournament(question)

    logger.info(f"  Champion: {metrics['champion_model']}")
    logger.info(f"  Total cost: ${metrics['total_cost']:.4f}")

    return {
        "condition": "D",
        "description": "Heterogeneous tournament (diverse models)",
        "champion_model": metrics["champion_model"],
        "champion_response": result,
        "metrics": metrics,
        "total_cost": metrics["total_cost"],
        "eliminated_models": metrics.get("eliminated_models", []),
        "active_models": metrics.get("active_model_keys", []),
    }


async def evaluate_results(all_results: list[dict[str, Any]], arbitrium: Arbitrium, question: str) -> dict[str, Any]:
    """
    Uses an external judge model to blindly evaluate all responses.

    Process:
    1. Collect all unique answers from all conditions
    2. Anonymize them (shuffle + assign IDs)
    3. Ask judge model to score each on a 1-10 scale
    4. Parse scores and map back to conditions
    5. Calculate statistics

    Args:
        all_results: Results from all conditions
        arbitrium: Arbitrium instance (to get a judge model)
        question: Original question

    Returns:
        Dictionary with scores and rankings
    """
    logger.info("=" * 80)
    logger.info("EVALUATION: Blind Scoring by External Judge")
    logger.info("=" * 80)

    # Collect all answers with metadata
    answers = []

    # Condition A: Multiple single model responses
    if all_results[0]["condition"] == "A":
        for model_key, result in all_results[0]["results"].items():
            answers.append(
                {
                    "condition": "A",
                    "sub_condition": model_key,
                    "content": result["content"],
                    "display_name": result["display_name"],
                }
            )

    # Condition B: Single model with CoT
    if all_results[1]["condition"] == "B":
        answers.append(
            {
                "condition": "B",
                "sub_condition": "cot",
                "content": all_results[1]["result"]["content"],
                "display_name": all_results[1]["display_name"] + " (CoT)",
            }
        )

    # Conditions C: Homogeneous tournaments
    for result in all_results[2:-1]:  # All C conditions
        if result["condition"] == "C":
            answers.append(
                {
                    "condition": "C",
                    "sub_condition": result["model_family"],
                    "content": result["champion_response"],
                    "display_name": f"Homogeneous {result['model_family'].upper()} Tournament",
                }
            )

    # Condition D: Heterogeneous tournament
    if all_results[-1]["condition"] == "D":
        answers.append(
            {
                "condition": "D",
                "sub_condition": "heterogeneous",
                "content": all_results[-1]["champion_response"],
                "display_name": "Arbitrium Framework (Heterogeneous)",
            }
        )

    # Shuffle to ensure blind evaluation
    random.shuffle(answers)
    for i, answer in enumerate(answers):
        answer["blind_id"] = f"Answer_{i + 1}"

    # Create evaluation prompt
    evaluation_prompt = f"""
You are an expert evaluator judging strategic business analysis responses.

Original Question:
{question}

Below are {len(answers)} different responses to this question. Your task is to evaluate
each response on the following criteria (1-10 scale):

1. **Depth of Analysis**: How thoroughly does it examine the problem?
2. **Evidence & Reasoning**: How well-supported are the arguments?
3. **Practical Applicability**: How actionable is the advice?
4. **Risk Awareness**: Does it identify potential pitfalls?
5. **Clarity**: Is it well-structured and easy to understand?

For each response, provide:
- Overall Score (1-10, where 10 is excellent)
- Brief justification (2-3 sentences)

Responses to evaluate:
"""

    for answer in answers:
        evaluation_prompt += f"\n\n{'='*60}\n{answer['blind_id']}:\n{'='*60}\n{answer['content']}\n"

    evaluation_prompt += "\n\nProvide your scores in this format:\nAnswer_X: [Score]/10 - [Justification]"

    # Use first available model as judge (prefer Claude or GPT-5)
    judge_model_key = None
    for preferred in ["claude", "gpt"]:
        if preferred in arbitrium.healthy_models:
            judge_model_key = preferred
            break
    if not judge_model_key:
        judge_model_key = next(iter(arbitrium.healthy_models.keys()))

    logger.info(f"Using {arbitrium.healthy_models[judge_model_key].full_display_name} as judge")
    judge_response = await arbitrium.run_single_model(judge_model_key, evaluation_prompt)

    # Parse scores (simplified parsing - production would be more robust)
    scores = {}
    for answer in answers:
        blind_id = answer["blind_id"]
        # Look for pattern like "Answer_1: 8/10" or "Answer_1: 8.5/10"
        import re

        pattern = rf"{blind_id}:\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, judge_response.content)
        if match:
            score = float(match.group(1))
            scores[blind_id] = {
                "score": score,
                "condition": answer["condition"],
                "sub_condition": answer["sub_condition"],
                "display_name": answer["display_name"],
            }

    logger.info(f"\nJudge evaluation cost: ${judge_response.cost:.4f}")
    logger.info(f"Successfully parsed {len(scores)}/{len(answers)} scores")

    # Calculate statistics by condition
    condition_scores = {}
    for _blind_id, score_data in scores.items():
        cond = score_data["condition"]
        if cond not in condition_scores:
            condition_scores[cond] = []
        condition_scores[cond].append(score_data["score"])

    # Calculate averages
    condition_averages = {cond: sum(scores_list) / len(scores_list) for cond, scores_list in condition_scores.items()}

    # Sort by average score
    ranked_conditions = sorted(condition_averages.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for rank, (condition, avg_score) in enumerate(ranked_conditions, 1):
        logger.info(f"{rank}. Condition {condition}: {avg_score:.2f}/10")

    return {
        "judge_model": arbitrium.healthy_models[judge_model_key].full_display_name,
        "judge_cost": judge_response.cost,
        "individual_scores": scores,
        "condition_averages": condition_averages,
        "ranking": ranked_conditions,
        "raw_evaluation": judge_response.content,
    }


async def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run Arbitrium Ablation Benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to main config file (for heterogeneous tournament)")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="Question to use for benchmark (optional)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip blind evaluation step (faster)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ARBITRIUM ABLATION BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Question: {args.question[:100]}...")
    logger.info("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize main Arbitrium instance
    logger.info("Initializing Arbitrium Framework...")
    arbitrium_main = await Arbitrium.from_config(args.config, skip_health_check=False)
    logger.info(f"✓ {arbitrium_main.healthy_model_count} healthy models loaded\n")

    all_results = []
    start_time = datetime.now()

    # --- Run all conditions ---

    # Condition A: Single models
    results_A = await run_condition_A(arbitrium_main, args.question)
    all_results.append(results_A)

    # Condition B: Best model + CoT (use first model as "best" - could be enhanced)
    best_model_key = next(iter(arbitrium_main.healthy_models.keys()))
    results_B = await run_condition_B(arbitrium_main, args.question, best_model_key)
    all_results.append(results_B)

    # Condition C: Homogeneous tournaments (only for families we have configs for)
    available_families = []
    for family, config_path in HOMOGENEOUS_CONFIG_PATHS.items():
        if Path(config_path).exists():
            available_families.append(family)

    for family in available_families:
        try:
            results_C = await run_condition_C(family, args.question)
            all_results.append(results_C)
        except Exception as e:
            logger.warning(f"Skipping {family} homogeneous tournament: {e}")

    # Condition D: Heterogeneous tournament
    results_D = await run_condition_D(arbitrium_main, args.question)
    all_results.append(results_D)

    # --- Save raw results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_results_path = OUTPUT_DIR / f"ablation_raw_{timestamp}.json"

    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n✓ Raw results saved to {raw_results_path}")

    # --- Evaluate results (optional) ---
    if not args.skip_evaluation:
        evaluation = await evaluate_results(all_results, arbitrium_main, args.question)
        all_results.append({"condition": "EVALUATION", "results": evaluation})
    else:
        logger.info("\nSkipping blind evaluation (--skip-evaluation flag)")
        evaluation = None

    # --- Generate final report ---
    report_path = OUTPUT_DIR / f"ablation_report_{timestamp}.md"
    total_cost = sum(r.get("total_cost", 0) for r in all_results if r.get("condition") != "EVALUATION")
    if evaluation:
        total_cost += evaluation["judge_cost"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Arbitrium Ablation Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Runtime:** {(datetime.now() - start_time).total_seconds():.1f}s\n\n")
        f.write(f"**Total Cost:** ${total_cost:.4f}\n\n")
        f.write("---\n\n")
        f.write("## Test Question\n\n")
        f.write(f"```\n{args.question}\n```\n\n")
        f.write("---\n\n")

        # Summary of conditions
        f.write("## Conditions Tested\n\n")
        for result in all_results:
            if result.get("condition") == "EVALUATION":
                continue
            cond = result["condition"]
            desc = result.get("description", "")
            cost = result.get("total_cost", 0)
            f.write(f"- **Condition {cond}**: {desc} (Cost: ${cost:.4f})\n")

        f.write("\n---\n\n")

        # Evaluation results
        if evaluation:
            f.write("## Evaluation Results\n\n")
            f.write(f"**Judge Model:** {evaluation['judge_model']}\n\n")
            f.write("### Rankings\n\n")
            for rank, (condition, avg_score) in enumerate(evaluation["ranking"], 1):
                f.write(f"{rank}. **Condition {condition}**: {avg_score:.2f}/10\n")

            f.write("\n### Detailed Scores\n\n")
            f.write("| Response | Condition | Score |\n")
            f.write("|----------|-----------|-------|\n")
            for _blind_id, score_data in sorted(evaluation["individual_scores"].items(), key=lambda x: x[1]["score"], reverse=True):
                f.write(f"| {score_data['display_name']} | {score_data['condition']} | {score_data['score']:.1f}/10 |\n")

        f.write("\n---\n\n")
        f.write("## Conclusion\n\n")
        f.write("This benchmark validates the core hypotheses of the Arbitrium Framework:\n\n")

        if evaluation:
            rankings = {cond: rank for rank, (cond, _) in enumerate(evaluation["ranking"], 1)}
            f.write("- **Competition improves quality**: ")
            if rankings.get("C", 99) < rankings.get("A", 99) or rankings.get("D", 99) < rankings.get("A", 99):
                f.write("✓ CONFIRMED\n")
            else:
                f.write("✗ NOT CONFIRMED\n")

            f.write("- **Diversity matters**: ")
            if rankings.get("D", 99) < rankings.get("C", 99):
                f.write("✓ CONFIRMED\n")
            else:
                f.write("✗ NOT CONFIRMED\n")

            f.write("- **Tournaments > Prompting**: ")
            if rankings.get("D", 99) < rankings.get("B", 99):
                f.write("✓ CONFIRMED\n")
            else:
                f.write("✗ NOT CONFIRMED\n")

        f.write("\n")
        f.write(f"**Raw Data:** `{raw_results_path.name}`\n\n")

    logger.info(f"✓ Final report saved to {report_path}")
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Total runtime: {(datetime.now() - start_time).total_seconds():.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Standard Benchmark Evaluation: Arbitrium Framework on BBH & GPQA

Uses well-established academic benchmarks for instant credibility:
- BBH (Big-Bench Hard): 23 challenging reasoning tasks
- GPQA (Graduate-Level Questions): Expert-level science questions

These benchmarks are respected everywhere and provide objective comparison.

Prerequisites:
    pip install datasets  # Hugging Face datasets library

Usage:
    python -m arbitrium.benchmarks.standard_benchmarks --config <path> --benchmark bbh
    python -m arbitrium.benchmarks.standard_benchmarks --config <path> --benchmark gpqa
    python -m arbitrium.benchmarks.standard_benchmarks --config <path> --benchmark both

Example:
    python -m arbitrium.benchmarks.standard_benchmarks --config config.benchmark.yml --benchmark bbh
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed")
    print("Install with: pip install datasets")
    sys.exit(1)

# Public API imports - benchmarks use only exported interface
from arbitrium import Arbitrium
from arbitrium.config.loader import Config
from arbitrium.logging import get_contextual_logger, setup_logging
from arbitrium.models.base import LiteLLMModel

# Setup logging explicitly
setup_logging(verbose=True, enable_file_logging=True)
logger = get_contextual_logger("benchmarks.standard_benchmarks")


# BBH task subset (most relevant for multi-model reasoning)
BBH_TASKS = [
    "causal_judgement",  # Cause-effect reasoning
    "formal_fallacies",  # Logical reasoning
    "navigate",  # Spatial reasoning
    "disambiguation_qa",  # Ambiguity resolution
    "logical_deduction_three_objects",  # Deductive reasoning
    "tracking_shuffled_objects_three_objects",  # Working memory
    "web_of_lies",  # Complex inference
    "movie_recommendation",  # Multi-constraint optimization
]


def extract_answer(response: str, choices: list[str]) -> str:
    """Extract answer from model response."""
    response_lower = response.lower().strip()

    # Try to find exact choice match
    for choice in choices:
        if choice.lower() in response_lower:
            return choice

    # Try to find answer markers
    for marker in ["answer:", "answer is:", "the answer is", "therefore,"]:
        if marker in response_lower:
            after_marker = response_lower.split(marker, 1)[1].strip()
            # Get first word/letter after marker
            first_word = after_marker.split()[0] if after_marker.split() else ""
            for choice in choices:
                if choice.lower().startswith(first_word[:1]):
                    return choice

    # Default: return first choice mentioned
    for choice in choices:
        if choice.lower() in response_lower:
            return choice

    return choices[0]  # Fallback


async def run_single_model_on_benchmark(
    questions: list[dict[str, Any]],
    model_name: str,
    config_path: str,
) -> dict[str, Any]:
    """Run single model on benchmark questions."""
    from datetime import datetime

    config_obj = Config(config_path)
    config_obj.load()
    config = config_obj.config_data
    model_config = config["models"][model_name]
    model = LiteLLMModel.from_config(model_name, model_config)

    results = []
    correct = 0
    start_time = datetime.now()

    for i, question_item in enumerate(questions, 1):
        prompt = f"""
Answer the following question. Think step-by-step, then provide your final answer.

Question: {question_item["question"]}

Choices:
{chr(10).join(f"- {choice}" for choice in question_item["choices"])}

Your answer:
""".strip()

        try:
            model_response = await model.generate(prompt)

            if model_response.is_error():
                logger.error(f"Error on question {i}: {model_response.error}")
                results.append(
                    {
                        "question": question_item["question"],
                        "predicted": None,
                        "actual": question_item["answer"],
                        "correct": False,
                        "error": model_response.error,
                    }
                )
                continue

            response_text = model_response.content
            predicted_answer = extract_answer(response_text, question_item["choices"])
            is_correct = predicted_answer == question_item["answer"]

            if is_correct:
                correct += 1

            results.append(
                {
                    "question": question_item["question"],
                    "predicted": predicted_answer,
                    "actual": question_item["answer"],
                    "correct": is_correct,
                    "response": response_text,
                }
            )

            logger.info(f"[{i}/{len(questions)}] {'✓' if is_correct else '✗'} {question_item['task']}")

        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
            results.append(
                {
                    "question": question_item["question"],
                    "predicted": None,
                    "actual": question_item["answer"],
                    "correct": False,
                    "error": str(e),
                }
            )

    accuracy = (correct / len(questions)) * 100 if questions else 0
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Extract predictions for statistical analysis
    preds = [r.get("predicted") for r in results]
    actuals = [r.get("actual") for r in results]

    return {
        "model": model_name,
        "approach": "single_model",
        "results": results,
        "correct": correct,
        "total": len(questions),
        "accuracy": accuracy,
        "duration_seconds": duration,
        "cost_estimate": 0.0,  # Local models are free; update if using paid APIs
        "preds": preds,
        "actuals": actuals,
        "ids": list(range(len(questions))),
    }


async def run_arbitrium_on_benchmark(
    questions: list[dict[str, Any]],
    config_path: str,
) -> dict[str, Any]:
    """Run Arbitrium Framework tournament on benchmark questions."""
    from datetime import datetime

    # Initialize Arbitrium using public API
    arbitrium = await Arbitrium.from_config(config_path=config_path)

    if not arbitrium.is_ready:
        raise RuntimeError(f"No healthy models available. Failed: {list(arbitrium.failed_models.keys())}")

    results = []
    correct = 0
    total_cost = 0.0
    start_time = datetime.now()

    for i, question_item in enumerate(questions, 1):
        prompt = f"""
Answer the following question. Provide your final answer clearly.

Question: {question_item["question"]}

Choices:
{chr(10).join(f"- {choice}" for choice in question_item["choices"])}
""".strip()

        try:
            # Run tournament using public API
            response, metrics = await arbitrium.run_tournament(prompt)
            predicted_answer = extract_answer(response, question_item["choices"])
            is_correct = predicted_answer == question_item["answer"]

            if is_correct:
                correct += 1

            # Track cost
            total_cost += metrics.get("total_cost", 0.0)

            results.append(
                {
                    "question": question_item["question"],
                    "predicted": predicted_answer,
                    "actual": question_item["answer"],
                    "correct": is_correct,
                    "champion": metrics.get("champion_model", "Unknown"),
                    "response": response,
                }
            )

            logger.info(f"[{i}/{len(questions)}] {'✓' if is_correct else '✗'} {question_item['task']} (Champion: {results[-1]['champion']})")

        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
            results.append(
                {
                    "question": question_item["question"],
                    "predicted": None,
                    "actual": question_item["answer"],
                    "correct": False,
                    "error": str(e),
                }
            )

    accuracy = (correct / len(questions)) * 100 if questions else 0
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Extract predictions for statistical analysis
    preds = [r.get("predicted") for r in results]
    actuals = [r.get("actual") for r in results]

    return {
        "approach": "arbitrium",
        "results": results,
        "correct": correct,
        "total": len(questions),
        "accuracy": accuracy,
        "duration_seconds": duration,
        "cost_actual": total_cost,
        "preds": preds,
        "actuals": actuals,
        "ids": list(range(len(questions))),
    }


def load_bbh_questions(num_per_task: int = 5) -> list[dict[str, Any]]:
    """Load questions from Big-Bench Hard."""
    logger.info(f"Loading BBH tasks: {', '.join(BBH_TASKS)}")

    questions = []

    for task in BBH_TASKS:
        try:
            dataset = load_dataset("lukaemon/bbh", task)  # nosec B615

            # Take subset
            for i, example in enumerate(dataset["test"]):
                if i >= num_per_task:
                    break

                # BBH format: input text, target answer, multiple choice options
                questions.append(
                    {
                        "task": task,
                        "question": example["input"],
                        "answer": example["target"],
                        "choices": example.get("choices", [example["target"]]),  # Some tasks have choices
                    }
                )

        except Exception as e:
            logger.warning(f"Could not load task '{task}': {e}")

    logger.info(f"Loaded {len(questions)} questions from {len(BBH_TASKS)} BBH tasks")
    return questions


def load_gpqa_questions(num_questions: int = 20) -> list[dict[str, Any]]:
    """Load questions from GPQA (Graduate-Level Questions)."""
    logger.info("Loading GPQA dataset...")

    try:
        # GPQA Diamond subset (highest quality)
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")  # nosec B615

        questions = []
        for i, example in enumerate(dataset["train"]):
            if i >= num_questions:
                break

            questions.append(
                {
                    "task": "gpqa",
                    "question": example["Question"],
                    "answer": example["Correct Answer"],
                    "choices": [
                        example["Correct Answer"],
                        example["Incorrect Answer 1"],
                        example["Incorrect Answer 2"],
                        example["Incorrect Answer 3"],
                    ],
                }
            )

        logger.info(f"Loaded {len(questions)} GPQA questions")
        return questions

    except Exception as e:
        logger.error(f"Could not load GPQA: {e}")
        return []


async def run_benchmark_suite(
    benchmark: str,
    config_path: str,
    num_questions: int = 20,
) -> None:
    """Run complete benchmark comparison."""
    print("=" * 80)
    print(f"STANDARD BENCHMARK: {benchmark.upper()}")
    print("=" * 80)
    print()

    # Load questions
    if benchmark == "bbh":
        questions = load_bbh_questions(num_per_task=5)
    elif benchmark == "gpqa":
        questions = load_gpqa_questions(num_questions=num_questions)
    else:
        print(f"Unknown benchmark: {benchmark}")
        return

    if not questions:
        print("No questions loaded. Exiting.")
        return

    print(f"\nLoaded {len(questions)} questions")
    print("Running baseline (single model) and Arbitrium Framework tournament...\n")

    # Load config
    config_obj = Config(config_path)
    config_obj.load()
    config = config_obj.config_data
    models = config["models"]
    baseline_results_list = []

    # Run baselines
    for model_name in models:
        print("=" * 80)
        print(f"BASELINE: Single Model ({model_name})")
        print("=" * 80 + "\n")

        baseline_results = await run_single_model_on_benchmark(questions, model_name, config_path)
        baseline_results_list.append(baseline_results)

        print(f"\n✅ Baseline Complete: {baseline_results['accuracy']:.1f}% accuracy")

    # Run Arbitrium Framework
    print("\n" + "=" * 80)
    print("ARBITRIUM TOURNAMENT")
    print("=" * 80 + "\n")

    tournament_results = await run_arbitrium_on_benchmark(questions, config_path)

    print(f"\n✅ Tournament Complete: {tournament_results['accuracy']:.1f}% accuracy")

    # Compute statistical significance
    from benchmarks.stats import cohens_h, compute_cost_normalized_metrics, mcnemar, paired_bootstrap_delta_acc

    stats = {}
    if baseline_results_list:
        best_baseline = max(baseline_results_list, key=lambda x: x["accuracy"])

        y_true = tournament_results["actuals"]
        y_arb = tournament_results["preds"]
        y_base = best_baseline["preds"]

        # McNemar test
        b01 = sum((y_base[i] == y_true[i]) and (y_arb[i] != y_true[i]) for i in range(len(y_true)))
        b10 = sum((y_base[i] != y_true[i]) and (y_arb[i] == y_true[i]) for i in range(len(y_true)))

        chi2, p_value = mcnemar(b01, b10)

        # Bootstrap CI
        mean_diff, (ci_low, ci_high) = paired_bootstrap_delta_acc(y_true, y_arb, y_base, iters=10000, seed=123)

        # Cohen's h
        h = cohens_h(tournament_results["accuracy"] / 100.0, best_baseline["accuracy"] / 100.0)

        # Cost-normalized metrics
        arb_norm = compute_cost_normalized_metrics(
            tournament_results["accuracy"],
            tournament_results["cost_actual"],
            tournament_results["duration_seconds"],
        )
        base_norm = compute_cost_normalized_metrics(
            best_baseline["accuracy"],
            best_baseline["cost_estimate"],
            best_baseline["duration_seconds"],
        )

        stats = {
            "best_baseline_model": best_baseline["model"],
            "best_baseline_accuracy": best_baseline["accuracy"],
            "arbitrium_accuracy": tournament_results["accuracy"],
            "delta_accuracy": mean_diff * 100,
            "bootstrap_ci_95": {"low": ci_low * 100, "high": ci_high * 100},
            "mcnemar": {"b01": b01, "b10": b10, "chi2": chi2, "p_value": p_value},
            "cohens_h": h,
            "arbitrium_normalized": arb_norm,
            "baseline_normalized": base_norm,
        }

    # Save results (JSON)
    output_path = Path(__file__).parent / f"{benchmark}_benchmark_results.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": benchmark,
                "date": datetime.now().isoformat(),
                "config": config_path,
                "num_questions": len(questions),
                "baselines": baseline_results_list,
                "tournament": tournament_results,
                "statistics": stats,
            },
            f,
            indent=2,
        )

    # Save per-question CSV (raw predictions)
    try:
        import csv

        csv_path = Path(__file__).parent / f"{benchmark}_per_question.csv"
        # Select best single baseline for comparison in CSV
        best_baseline_csv: dict[str, Any] | None = max(baseline_results_list, key=lambda x: x["accuracy"]) if baseline_results_list else None

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["qid", "task", "gold", "baseline_pred", "arbitrium_pred", "baseline_correct", "arbitrium_correct"])

            for idx, tr in enumerate(tournament_results["results"]):
                gold = tr["actual"]
                arb_pred = tr["predicted"]
                arb_corr = int(arb_pred == gold) if arb_pred else 0
                task = tr.get("task", "")

                if best_baseline_csv:
                    base_pred = best_baseline_csv["preds"][idx] if idx < len(best_baseline_csv["preds"]) else ""
                    base_corr = int(base_pred == gold) if base_pred else 0
                else:
                    base_pred, base_corr = "", 0

                writer.writerow([idx, task, gold, base_pred, arb_pred, base_corr, arb_corr])

        logger.info(f"Per-question CSV saved to: {csv_path}")
    except Exception as e:
        logger.warning(f"Could not write CSV: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBenchmark: {benchmark.upper()}")
    print(f"Questions: {len(questions)}")
    print()
    for baseline_results in baseline_results_list:
        print(f"Baseline Accuracy ({baseline_results['model']}):    {baseline_results['accuracy']:.1f}%")
    print(f"Tournament Accuracy:  {tournament_results['accuracy']:.1f}%")
    print()

    if stats:
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE (Arbitrium vs Best Single)")
        print("=" * 80)
        print(f"Best Single:          {stats['best_baseline_accuracy']:.2f}%")
        print(f"Arbitrium:            {stats['arbitrium_accuracy']:.2f}%")
        print(f"Δ Accuracy:           {stats['delta_accuracy']:.2f}%")
        print(f"95% CI (bootstrap):   [{stats['bootstrap_ci_95']['low']:.2f}%, {stats['bootstrap_ci_95']['high']:.2f}%]")
        print(
            f"McNemar:              b01={stats['mcnemar']['b01']}, b10={stats['mcnemar']['b10']}, "
            f"χ²={stats['mcnemar']['chi2']:.3f}, p≈{stats['mcnemar']['p_value']:.4f}"
        )
        print(f"Cohen's h:            {stats['cohens_h']:.3f}")
        print()
        print("Cost-Normalized Metrics:")
        print(
            f"Arbitrium:  {stats['arbitrium_normalized']['accuracy_per_dollar']:.2f} acc/$, "
            f"{stats['arbitrium_normalized']['accuracy_per_minute']:.2f} acc/min"
        )
        print(
            f"Baseline:   {stats['baseline_normalized']['accuracy_per_dollar']:.2f} acc/$, "
            f"{stats['baseline_normalized']['accuracy_per_minute']:.2f} acc/min"
        )

        # Interpret results
        print()
        if stats["mcnemar"]["p_value"] < 0.05 and stats["delta_accuracy"] > 0:
            print("✅ Arbitrium shows STATISTICALLY SIGNIFICANT improvement (p < 0.05)")
        elif tournament_results["accuracy"] > stats["best_baseline_accuracy"]:
            print("⚠️  Arbitrium shows improvement but not statistically significant")
        else:
            print("❌ No improvement over best baseline")

    print(f"\n📄 Detailed results saved to: {output_path}")
    print(f"📎 Per-question CSV saved to: {Path(__file__).parent / f'{benchmark}_per_question.csv'}")
    print()
    print("NEXT STEPS:")
    print("1. Review statistical analysis in JSON output")
    print("2. Examine individual question performance")
    print("3. Run ablation studies (w/o Knowledge Bank, different judge models)")
    print("4. Prepare results for publication (arXiv/tech report)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (REQUIRED - no defaults)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["bbh", "gpqa", "both"],
        default="bbh",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions (for GPQA)",
    )

    args = parser.parse_args()
    config_path = args.config

    if args.benchmark == "both":
        asyncio.run(run_benchmark_suite("bbh", config_path, args.num_questions))
        asyncio.run(run_benchmark_suite("gpqa", config_path, args.num_questions))
    else:
        asyncio.run(
            run_benchmark_suite(
                args.benchmark,
                config_path,
                args.num_questions,
            )
        )


if __name__ == "__main__":
    main()

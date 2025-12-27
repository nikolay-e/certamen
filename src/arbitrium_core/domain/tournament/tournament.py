"Core comparison functionality for Arbitrium Framework."

import asyncio
import logging
import re
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arbitrium_core.shared.logging.structured import ContextualLogger

from arbitrium_core.domain.errors import (
    BudgetExceededError,
    TournamentTimeoutError,
)
from arbitrium_core.domain.knowledge.bank import EnhancedKnowledgeBank
from arbitrium_core.domain.model_selection import select_model_by_capacity
from arbitrium_core.domain.prompts import (
    LOG_EVALUATOR_RESPONSE,
    PromptBuilder,
    PromptFormatter,
)
from arbitrium_core.domain.tournament.anonymizer import ModelAnonymizer
from arbitrium_core.domain.tournament.report import ReportGenerator
from arbitrium_core.domain.tournament.scoring import ScoreExtractor
from arbitrium_core.ports.llm import BaseModel, ModelResponse
from arbitrium_core.ports.similarity import SimilarityEngine
from arbitrium_core.shared.constants import (
    PLACEHOLDER_RESPONSES,
    VARIANCE_HIGH_CONFIDENCE_THRESHOLD,
    VARIANCE_MEDIUM_CONFIDENCE_THRESHOLD,
)
from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.text import indent_text, strip_meta_commentary


# Internal interfaces for ModelComparison
class EventHandler(ABC):
    @abstractmethod
    def publish(self, _event_name: str, _data: dict[str, Any]) -> None:
        pass


class HostEnvironment(ABC):
    base_dir: Path | str  # Output directory path (required by implementations)

    @abstractmethod
    async def read_file(self, path: str) -> str:
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        pass

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        pass


class BudgetGuard:
    """Guards against excessive tournament costs and runtime.

    Monitors tournament spending and elapsed time, raising exceptions
    when configured limits are exceeded. This prevents runaway costs
    and ensures tournaments complete within reasonable timeframes.

    Args:
        max_cost: Maximum allowed cost in USD (default: 5.0)
        max_time: Maximum allowed time in seconds (default: 900 = 15 minutes)

    Example:
        ```python
        guard = BudgetGuard(max_cost=5.0, max_time=600)
        guard.check(spent=2.5, elapsed=300)  # OK
        guard.check(spent=6.0, elapsed=300)  # Raises BudgetExceededError
        ```
    """

    def __init__(self, max_cost: float = 5.0, max_time: float = 900.0) -> None:
        self.budget = max_cost
        self.timeout = max_time
        self.start_time = datetime.now()

    def check(self, spent: float, elapsed: float | None = None) -> None:
        """Check if budget or time limits have been exceeded.

        Args:
            spent: Total cost spent so far in USD
            elapsed: Elapsed time in seconds (optional, auto-calculated if None)

        Raises:
            BudgetExceededError: If spent >= max_cost
            TournamentTimeoutError: If elapsed >= max_time
        """
        # Check budget
        if spent >= self.budget:
            raise BudgetExceededError(
                "Tournament stopped to prevent further costs.",
                spent=spent,
                budget=self.budget,
            )

        # Check timeout
        if elapsed is None:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        if elapsed >= self.timeout:
            raise TournamentTimeoutError(
                "Tournament stopped due to time limit.",
                elapsed=elapsed,
                timeout=self.timeout,
            )

    def reset(self) -> None:
        self.start_time = datetime.now()


class CostTracker:
    def __init__(
        self, logger: "logging.Logger | ContextualLogger | None" = None
    ):
        self.total_cost = 0.0
        self.cost_by_model: dict[str, float] = {}
        self.logger = logger or get_contextual_logger("arbitrium.cost_tracker")

    def add_cost(self, model_name: str, cost: float) -> None:
        self.total_cost += cost

        if model_name not in self.cost_by_model:
            self.cost_by_model[model_name] = 0.0
        self.cost_by_model[model_name] += cost

        self.logger.debug(
            "Added $%.4f for %s, total now: $%.4f",
            cost,
            model_name,
            self.total_cost,
        )

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_cost": f"${self.total_cost:.4f}",
            "cost_by_model": {
                k: f"${v:.4f}" for k, v in self.cost_by_model.items()
            },
        }

    def display_summary(self) -> None:
        self.logger.info(
            "Cost Summary", extra={"display_type": "section_header"}
        )
        self.logger.info(
            f"Total Cost: ${self.total_cost:.4f}",
            extra={"display_type": "colored_text"},
        )

        if self.cost_by_model:
            self.logger.info(
                "\nCost by Model:", extra={"display_type": "colored_text"}
            )
            for model_name, cost in sorted(
                self.cost_by_model.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (
                    (cost / self.total_cost) * 100
                    if self.total_cost > 0
                    else 0
                )
                self.logger.info(
                    f"  {model_name}: ${cost:.4f} ({percentage:.1f}%)",
                    extra={"display_type": "colored_text"},
                )
        else:
            self.logger.info(
                "No cost information available (cost tracking may be disabled)",
                extra={"display_type": "colored_text"},
            )

        self.logger.info("Tournament total cost: $%.4f", self.total_cost)


class TournamentRunner:
    def __init__(self, comparison_instance: "ModelComparison") -> None:
        self.comp = comparison_instance
        self.event_handler = comparison_instance.event_handler
        self.logger = comparison_instance.logger

    async def _run_tournament_phases(self, initial_question: str) -> str:
        if not await self._run_initial_phase(initial_question):
            return "No valid initial responses. Tournament cannot proceed."
        if not await self._run_phase_2(initial_question):
            return "Phase 2 failed. Tournament cannot proceed."
        await self._run_elimination_rounds(initial_question)
        return await self._finalize_tournament(initial_question)

    async def run(self, initial_question: str) -> str:
        # Set run_id for the entire tournament
        self.logger.set_run()
        self.logger.info(
            "Starting model comparison tournament", question=initial_question
        )
        self.comp.previous_answers = []
        self.comp.eliminated_models = []
        self.comp.evaluation_history = []
        self.comp.feedback_history = []
        self.comp.criticism_history = []

        try:
            return await self._run_tournament_phases(initial_question)
        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user.")
            return "Process interrupted by user."
        except Exception as e:
            self.logger.exception("Unexpected error in tournament: %s", e)
            return f"Tournament error: {e!s}"

    async def _run_initial_phase(self, initial_question: str) -> bool:
        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            "PHASE 1: Initial Answers - Each model answers independently"
        )
        self.logger.info("=" * 80)
        initial_responses = await self.comp.run_initial_round(initial_question)
        if not initial_responses:
            return False
        self.comp.previous_answers.append(initial_responses)
        self.logger.info(
            f"PHASE 1 COMPLETE: Got {len(initial_responses)} initial responses"
        )
        return True

    async def _run_improvement_phase(
        self,
        initial_question: str,
        config_key: str,
        phase_name: str,
        answer_index: int,
        round_num: int | None = None,
    ) -> bool:
        phase_config = self.comp.config.get(config_key, {})

        if not phase_config.get("enabled", True):
            skip_msg = f"{phase_name} is disabled. Skipping."
            if round_num is not None:
                skip_msg = (
                    f"{phase_name} is disabled. Skipping round {round_num}."
                )
            self.logger.info(skip_msg)
            return True

        if round_num is None:
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"{phase_name}")
            self.logger.info("=" * 80)
        else:
            self.logger.info(f"\nROUND {round_num}: {phase_name}")

        source_answers = self.comp.previous_answers[answer_index]

        feedback_context: dict[str, dict[str, str]] | None = None
        if phase_config.get("feedback_enabled", False):
            self.logger.info("Collecting feedback from models...")
            feedback_context = await self.comp.run_feedback(
                initial_question,
                source_answers,
                feedback_instruction=phase_config.get(
                    "feedback_instruction", "Provide feedback for this answer."
                ),
            )
            if not feedback_context:
                self.logger.warning(
                    "No feedback collected, proceeding without it"
                )
                feedback_context = None

        action_word = "refined" if round_num is not None else "improved"
        self.logger.info(f"Generating {action_word} responses...")

        default_instruction = (
            "Refine your answer." if round_num else "Improve your answer."
        )
        responses = await self.comp.run_improvement(
            initial_question,
            source_answers,
            improvement_instruction=phase_config.get(
                "improvement_instruction", default_instruction
            ),
            improvement_context=feedback_context,
            other_responses=(
                source_answers
                if phase_config.get("share_responses", True)
                else None
            ),
        )

        if not responses:
            return False

        self.comp.previous_answers.append(responses)
        if round_num is None:
            self.logger.info(
                f"{phase_name} COMPLETE: Got {len(responses)} {action_word} responses"
            )
        else:
            self.logger.info(
                f"ROUND {round_num} COMPLETE: Got {len(responses)} {action_word} responses"
            )
        return True

    async def _run_phase_2(self, initial_question: str) -> bool:
        return await self._run_improvement_phase(
            initial_question,
            config_key="improvement_phase",
            phase_name="PHASE 2: Improvement Phase",
            answer_index=0,
        )

    async def _run_elimination_rounds(self, initial_question: str) -> None:
        round_num = 1
        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            f"Starting elimination rounds with {len(self.comp.active_model_keys)} models"
        )
        self.logger.info("=" * 80)

        while len(self.comp.active_model_keys) > 1:
            self.logger.info("\n" + "-" * 80)
            self.logger.info(f"ROUND {round_num}: Cross-Evaluation Phase")
            self.logger.info("-" * 80)

            evaluations = await self.comp.run_cross_evaluation(
                initial_question, self.comp.previous_answers[-1], round_num
            )
            if not evaluations:
                # Check if we still have models after evaluation failures
                if len(self.comp.active_model_keys) == 0:
                    self.logger.error(
                        f"No evaluations in round {round_num} and no active models remain. Tournament failed."
                    )
                    break
                elif len(self.comp.active_model_keys) == 1:
                    self.logger.warning(
                        f"No evaluations in round {round_num}, but 1 model remains. Declaring champion."
                    )
                    break
                else:
                    self.logger.warning(
                        f"No evaluations in round {round_num}, but {len(self.comp.active_model_keys)} models remain. "
                        f"This indicates an evaluation system failure. Declaring current leader as champion."
                    )
                    break

            self.comp.evaluation_history.append(
                {
                    "round": round_num,
                    "evaluations": evaluations.copy(),
                    "scores": (
                        self.comp.evaluation_scores.copy()
                        if hasattr(self.comp, "evaluation_scores")
                        else {}
                    ),
                }
            )

            (
                eliminated_model,
                _leader_model,
            ) = self.comp.determine_lowest_and_highest_ranked_models()
            if not eliminated_model:
                self.logger.warning(
                    "Could not determine model to eliminate. Ending tournament."
                )
                break

            await self._handle_elimination(eliminated_model, round_num)

            if len(self.comp.active_model_keys) <= 1:
                break

            if not await self._run_refinement_strategy(
                initial_question, round_num
            ):
                break

            round_num += 1

    async def _run_refinement_strategy(
        self,
        initial_question: str,
        round_num: int,
    ) -> bool:
        return await self._run_improvement_phase(
            initial_question,
            config_key="refinement_phase",
            phase_name="Refinement Phase",
            answer_index=-1,
            round_num=round_num,
        )

    async def _handle_elimination(
        self, eliminated_model: str, round_num: int
    ) -> None:
        eliminated_response = self.comp.previous_answers[-1].get(
            eliminated_model
        )

        # Extract insights from eliminated model
        insights_preserved: list[str] = []
        if eliminated_response:
            await self.comp.knowledge_bank.extract_and_add_insights(
                eliminated_response, eliminated_model, round_num
            )
            # Retrieve the insights that were just added
            insights_preserved = (
                await self.comp.knowledge_bank.get_insights_for_model(
                    eliminated_model, round_num
                )
            )
            self.logger.info(
                f"Preserved {len(insights_preserved)} insights from {eliminated_model}"
            )

        # Get variance and consensus data for this elimination
        score_variance = None
        elimination_confidence = "unknown"
        if hasattr(self.comp, "model_score_variances"):
            variance_data = self.comp.model_score_variances.get(
                eliminated_model
            )
            if variance_data:
                score_variance = variance_data["stdev"]
                elimination_confidence = (
                    self._determine_elimination_confidence(score_variance)
                )

        # Store elimination info with reason and score for provenance
        elimination_info: dict[str, object] = {
            "model": eliminated_model,
            "round": round_num,
            "reason": getattr(
                self.comp, "elimination_reason", "Lowest score in evaluation"
            ),
            "score": getattr(self.comp, "elimination_score", None),
            "score_variance": score_variance,
            "elimination_confidence": elimination_confidence,
            "insights_preserved": insights_preserved,  # Now populated!
        }

        # Add self-scoring bias if available
        if hasattr(self.comp, "self_scoring_biases"):
            bias = self.comp.self_scoring_biases.get(eliminated_model)
            if bias is not None:
                elimination_info["self_scoring_bias"] = round(bias, 2)

        self.comp.eliminated_models.append(elimination_info)

        model_key_to_remove = next(
            (
                key
                for key, name in self.comp.anon_mapping.items()
                if name == eliminated_model
            ),
            None,
        )

        if model_key_to_remove:
            self.comp.active_model_keys.remove(model_key_to_remove)
            self.comp.anon_mapping.pop(model_key_to_remove)

        self.logger.info(
            f"ELIMINATED: {eliminated_model} - {len(self.comp.active_model_keys)} models remaining"
        )
        self.logger.info(f"   Reason: {elimination_info['reason']}")

    def _determine_elimination_confidence(self, score_variance: float) -> str:
        if score_variance < VARIANCE_HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        if score_variance < VARIANCE_MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"

    async def _finalize_tournament(self, initial_question: str) -> str:
        if len(self.comp.active_model_keys) == 0:
            self.logger.error(
                "All models failed during tournament. No champion can be determined."
            )
            return "Tournament ended prematurely: All models failed or were eliminated due to errors."
        elif len(self.comp.active_model_keys) == 1:
            final_model_key = self.comp.active_model_keys[0]
            final_model_anon = self.comp.anon_mapping[final_model_key]

            champion_answer = self.comp.previous_answers[-1].get(
                final_model_anon, ""
            )

            if not champion_answer:
                self.logger.error(
                    f"Could not find final answer for champion {final_model_anon}"
                )
                return f"Champion {final_model_anon} determined but answer not found."

            self.logger.info(
                f"CHAMPION: {final_model_anon} - Using their final refined answer"
            )

            if self.comp.features.get("save_reports_to_disk", True):
                await self.comp._save_champion_report(
                    initial_question=initial_question,
                    final_model_anon=final_model_anon,
                    champion_answer=champion_answer,
                    all_previous_answers=self.comp.previous_answers,
                )

            self.logger.info(
                "Champion's Final Answer",
                extra={"display_type": "section_header"},
            )
            self.logger.info(
                champion_answer,
                extra={
                    "display_type": "model_response",
                    "model_name": "success",
                },
            )

            return champion_answer
        else:
            msg = f"Tournament ended with {len(self.comp.active_model_keys)} models remaining. No single champion determined."
            self.logger.warning(
                f"Tournament ended with {len(self.comp.active_model_keys)} models remaining (no single champion)."
            )
            return msg


class ModelComparison:
    def __init__(
        self,
        config: dict[str, Any],
        models: dict[str, BaseModel],
        event_handler: EventHandler,
        host: HostEnvironment,
        similarity_engine: "SimilarityEngine",
    ):
        self.config = config
        self.models = models
        self.event_handler = event_handler
        self.host = host
        self.similarity_engine = similarity_engine

        # Initialize contextual logger for correlation IDs
        self.logger = get_contextual_logger("arbitrium.comparison")

        self.cost_tracker = CostTracker(logger=self.logger)

        self.retry_settings = config["retry"]
        self.features = config["features"]
        self.prompts = config["prompts"]

        self.previous_answers: list[dict[str, str]] = []
        self.eliminated_models: list[dict[str, Any]] = []
        self.evaluation_history: list[dict[str, Any]] = []
        self.evaluation_scores: (
            dict[str, float] | dict[str, dict[str, float]]
        ) = {}
        self.all_evaluations: dict[str, str] = {}
        self.feedback_history: list[dict[str, Any]] = []
        self.criticism_history: list[dict[str, Any]] = []

        # Elimination tracking attributes (set during determine_lowest_and_highest_ranked_models)
        self.elimination_reason: str = ""
        self.elimination_score: float | None = None

        deterministic_mode = self.features.get("deterministic_mode", False)
        if deterministic_mode:
            self.logger.info(
                "Running in deterministic mode with fixed random seed"
            )

        self.anonymizer = ModelAnonymizer(deterministic_mode)
        self.score_extractor = ScoreExtractor()
        self.report_generator = ReportGenerator(self.host)
        self.formatter = PromptFormatter()
        self.prompt_builder = PromptBuilder(self.prompts, self.formatter)

        self.active_model_keys = list(models.keys())
        self.judge_model_key = self._identify_and_remove_judge()

        self.anon_mapping = self.anonymizer.anonymize_model_keys(
            self.active_model_keys
        )

        self.knowledge_bank = EnhancedKnowledgeBank(
            self, self.similarity_engine
        )

        max_concurrent = (
            config.get("model_defaults", {})
            .get("concurrency_limits", {})
            .get("max_concurrent_requests", 2)
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.runner = TournamentRunner(self)

    @property
    def total_cost(self) -> float:
        return self.cost_tracker.total_cost

    @property
    def cost_by_model(self) -> dict[str, float]:
        return self.cost_tracker.cost_by_model

    def _identify_and_remove_judge(self) -> str | None:
        judge_model_config_key = self.features.get("judge_model")
        if not judge_model_config_key:
            return None

        judge_model_key = None
        for key, model_instance in self.models.items():
            if (
                key == judge_model_config_key
                or model_instance.display_name == judge_model_config_key
            ):
                judge_model_key = key
                break

        if not judge_model_key:
            self.logger.warning(
                f"Judge model '{judge_model_config_key}' not found in available models"
            )
            return None

        if judge_model_key in self.active_model_keys:
            self.logger.info(
                f"Judge model '{self.models[judge_model_key].display_name}' will only act as judge and will not participate in the tournament.",
                extra={"display_type": "colored_text", "color": "warning"},
            )
            self.active_model_keys.remove(judge_model_key)
            self.logger.info(
                f"Removed judge model {judge_model_key} from tournament participants"
            )

        return judge_model_key

    async def _get_knowledge_bank_context(self) -> str:
        self.config.get("knowledge_bank", {})
        # User pays for full context - return ALL insights, no limits
        return await self.knowledge_bank.format_insights_for_context()

    def _filter_valid_responses(
        self,
        results: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        valid = {}
        failed_keys = []
        for key, value in results.items():
            if not value:
                failed_keys.append(key)
                continue
            txt = value.strip()
            if not txt or txt.lower().startswith("error:"):
                failed_keys.append(key)
                continue

            # Check for placeholder responses (case-insensitive)
            if len(txt) < 10 or txt.strip().lower() in PLACEHOLDER_RESPONSES:
                self.logger.warning(
                    f"Filtered out placeholder/invalid response from {key}: '{txt}'"
                )
                failed_keys.append(key)
                continue

            valid[key] = txt
        return valid, failed_keys

    def _decode_shuffled_names(
        self,
        text: str,
        reverse_shuffle_mapping: dict[str, str],
    ) -> str:
        self.logger.debug(
            f"Decoding shuffled names: {len(reverse_shuffle_mapping)} mappings"
        )
        decoded_text = text
        for code_name, orig_name in reverse_shuffle_mapping.items():
            pattern = r"\b" + re.escape(code_name) + r"\b"
            decoded_text = re.sub(pattern, f"({orig_name})", decoded_text)
            self.logger.debug(f"Decoded: {code_name} → {orig_name}")
        return decoded_text

    def _display_model_score(
        self,
        model_name: str,
        score: float,
        score_type: str = "Score",
    ) -> None:
        self.logger.info(
            f"{model_name}: {score_type} {score:.2f}/10",
            extra={"display_type": "colored_text", "color": model_name},
        )

    def _handle_model_failure(self, model_key: str, reason: str) -> None:
        display_name = self.anon_mapping.get(model_key, model_key)
        self.logger.error(f"Removing {display_name} from tournament: {reason}")

        if model_key in self.active_model_keys:
            self.active_model_keys.remove(model_key)
        self.anon_mapping.pop(model_key, None)

    async def _execute_single_model_task(
        self,
        model_key: str,
        prompt: str,
        context_for_logging: str,
    ) -> ModelResponse:
        from arbitrium_core.shared.constants import DEFAULT_MODEL_TIMEOUT

        model = self.models[model_key]
        self.anon_mapping.get(model_key, model.display_name)

        # Use task context for correlation IDs
        with self.logger.task_context(
            phase=context_for_logging, model=model_key
        ):
            self.logger.debug(
                f"Preparing to execute task for model: {model_key}"
            )

            self.logger.debug(
                f"Executing {context_for_logging}",
                model=model.display_name,
                model_id=model.model_name,
            )

            # Log the full prompt at DEBUG level
            log_message = self.formatter.format_log_message(
                "PROMPT", model.display_name, prompt
            )
            self.logger.debug(indent_text(log_message))

            try:
                async with self.semaphore:
                    timeout_value = DEFAULT_MODEL_TIMEOUT * 4
                    self.logger.debug(
                        f"Starting async request with {timeout_value}s timeout"
                    )
                    response = await asyncio.wait_for(
                        model.generate_with_retry(
                            prompt=prompt,
                            max_attempts=self.retry_settings.get(
                                "max_attempts", 3
                            ),
                            initial_delay=None,
                            max_delay=None,
                        ),
                        timeout=timeout_value,
                    )
            except TimeoutError:
                self.logger.error(
                    f"Task timeout for {model.full_display_name} after {DEFAULT_MODEL_TIMEOUT * 4}s"
                )
                return ModelResponse.create_error(
                    f"Task timeout after {DEFAULT_MODEL_TIMEOUT * 4} seconds"
                )

            if hasattr(response, "cost"):
                if response.cost > 0:
                    model_display_name = model.display_name
                    self.cost_tracker.add_cost(
                        model_display_name, response.cost
                    )
                    self.logger.info(
                        f"Added ${response.cost:.4f} for {model_display_name}, total now: ${self.cost_tracker.total_cost:.4f}"
                    )
                else:
                    self.logger.debug(
                        f"Zero cost response from {model.display_name}"
                    )
            else:
                self.logger.debug(
                    f"No cost attribute in response from {model.display_name}"
                )

            # Apply meta-commentary filtering for improvement responses
            if (
                response
                and response.content
                and context_for_logging == "IMPROVEMENT"
            ):
                cleaned_content = strip_meta_commentary(
                    response.content, logger=self.logger
                )
                # Update the response object with cleaned content
                response.content = cleaned_content

            # Log response immediately to ensure it's saved even if subsequent tasks fail
            if response and response.content and context_for_logging:
                display_name = self.anon_mapping.get(
                    model_key, model.display_name
                )
                # Format as [RESPONSE_TYPE] FROM model
                label = f"[{context_for_logging.upper()}] RESPONSE FROM {display_name}"
                log_message = self.formatter.wrap_section(
                    label, response.content
                )
                self.logger.info(indent_text(log_message))

            return response

    def _prepare_parallel_tasks(
        self,
        model_keys_to_run: list[str],
        prompt_builder: Callable[[str, BaseModel], str],
        context_for_logging: str,
    ) -> tuple[list[tuple[str, Any]], dict[str, str]]:
        tasks = []
        display_names = {}

        for model_key in model_keys_to_run:
            if model_key not in self.active_model_keys:
                continue

            model = self.models[model_key]
            display_name = self.anon_mapping[model_key]
            display_names[model_key] = display_name

            prompt = prompt_builder(model_key, model)
            task = self._execute_single_model_task(
                model_key=model_key,
                prompt=prompt,
                context_for_logging=context_for_logging,
            )
            tasks.append((model_key, task))

        return tasks, display_names

    def _process_single_response(
        self,
        response: Any,
        model_key: str,
        display_name: str,
        context_for_logging: str,
        results: dict[str, str],
    ) -> None:
        if isinstance(response, Exception):
            self._handle_model_failure(
                model_key, f"Error in {context_for_logging}: {response!s}"
            )
            return

        if not isinstance(response, ModelResponse):
            return

        if response.is_error():
            error_msg = response.error or "Unknown error"
            self._handle_model_failure(
                model_key, f"Error in {context_for_logging}: {error_msg}"
            )
            return

        # Content is already cleaned by _execute_single_model_task if needed
        results[display_name] = response.content

    def _handle_failed_models(self, failed_keys: list[str]) -> None:
        for display_name in failed_keys:
            model_key_to_remove = next(
                (k for k, v in self.anon_mapping.items() if v == display_name),
                None,
            )
            if model_key_to_remove:
                self._handle_model_failure(
                    model_key_to_remove, "Invalid/empty response"
                )

    def _check_tournament_viability(
        self,
        valid_results: dict[str, str],
        context_for_logging: str,
    ) -> bool:
        # For PEER_EVAL, allow fallback to judge mode even if no valid results
        # The evaluation phase will handle fallback logic
        if context_for_logging == "PEER_EVAL":
            if not valid_results:
                self.logger.warning(
                    "No valid peer evaluation responses. Will try judge fallback."
                )
            return True

        if not valid_results:
            self.logger.critical(
                f"No valid responses for {context_for_logging} from any models."
            )
            self.logger.warning(
                f"Tournament cannot continue. Active models remaining: {len(self.active_model_keys)}"
            )
            return False

        if len(valid_results) < 2 and context_for_logging in [
            "INITIAL",
            "IMPROVEMENT",
        ]:
            self.logger.warning(
                f"Only {len(valid_results)} model(s) responded. Tournament may end prematurely."
            )

        return True

    async def _execute_parallel_model_tasks(
        self,
        model_keys_to_run: list[str],
        prompt_builder: Callable[[str, BaseModel], str],
        context_for_logging: str,
    ) -> dict[str, str]:
        tasks, display_names = self._prepare_parallel_tasks(
            model_keys_to_run, prompt_builder, context_for_logging
        )
        results: dict[str, str] = {}

        try:
            self.logger.info(
                f"Gathering {len(tasks)} parallel {context_for_logging} tasks"
            )
            responses = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            for i, response in enumerate(responses):
                model_key = tasks[i][0]
                display_name = display_names[model_key]
                self._process_single_response(
                    response,
                    model_key,
                    display_name,
                    context_for_logging,
                    results,
                )
        except Exception as e:
            self.logger.error(
                f"Unexpected error during parallel {context_for_logging} calls: {e!s}"
            )

        valid_results, failed_keys = self._filter_valid_responses(results)
        self._handle_failed_models(failed_keys)

        if not self._check_tournament_viability(
            valid_results, context_for_logging
        ):
            return {}

        return valid_results

    async def run_initial_round(self, initial_question: str) -> dict[str, str]:
        self.logger.info(
            "Individual Response Generation",
            extra={"display_type": "section_header"},
        )

        def build_initial_prompt(model_key: str, model: BaseModel) -> str:
            return self.prompt_builder.build_initial_prompt(initial_question)

        valid_responses = await self._execute_parallel_model_tasks(
            model_keys_to_run=self.active_model_keys,
            prompt_builder=build_initial_prompt,
            context_for_logging="INITIAL",
        )

        if not valid_responses:
            self.logger.error("No valid responses from any models.")
            return {}

        self.logger.info(
            "Initial Model Responses", extra={"display_type": "section_header"}
        )

        return valid_responses

    async def run_feedback(
        self,
        initial_question: str,
        current_responses: dict[str, str],
        feedback_instruction: str,
    ) -> dict[str, dict[str, str]]:
        active_responses = {
            name: resp
            for name, resp in current_responses.items()
            if name in [self.anon_mapping[k] for k in self.active_model_keys]
        }

        feedback_context: dict[str, dict[str, str]] = {
            model_name: {} for model_name in active_responses.keys()
        }

        tasks = []
        task_metadata = []

        for target_model, target_answer in active_responses.items():
            reviewer_models = [
                m
                for m in self.active_model_keys
                if self.anon_mapping[m] != target_model
            ]

            for reviewer_key in reviewer_models:
                prompt = self.prompt_builder.build_feedback_prompt(
                    initial_question=initial_question,
                    target_answer=target_answer,
                    feedback_instruction=feedback_instruction,
                )
                task = self._execute_single_model_task(
                    model_key=reviewer_key,
                    prompt=prompt,
                    context_for_logging="FEEDBACK",
                )
                tasks.append(task)
                task_metadata.append((reviewer_key, target_model))

        if tasks:
            self.logger.info(f"Gathering {len(tasks)} parallel feedback tasks")
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                reviewer_key, target_model = task_metadata[i]
                reviewer_name = self.anon_mapping[reviewer_key]

                if isinstance(response, Exception):
                    self.logger.error(
                        f"Error getting feedback from {reviewer_name} for {target_model}: {response!s}"
                    )
                    continue

                if not isinstance(response, ModelResponse):
                    continue

                if response.is_error():
                    self.logger.error(
                        f"Error getting feedback from {reviewer_name} for {target_model}: {response.error}"
                    )
                    continue

                feedback_text = response.content
                feedback_context[target_model][reviewer_name] = feedback_text

                log_message = self.formatter.format_feedback_log(
                    reviewer_name, target_model, feedback_text
                )
                self.logger.debug(indent_text(log_message))
                self.logger.info(
                    f"\n{reviewer_name}'s feedback for {target_model}:",
                    extra={
                        "display_type": "colored_text",
                        "color": reviewer_name,
                    },
                )
                self.logger.info(
                    feedback_text,
                    extra={
                        "display_type": "colored_text",
                        "color": reviewer_name,
                    },
                )
                self.logger.info(
                    "-" * 20, extra={"display_type": "colored_text"}
                )

        self.feedback_history.append(
            {"round": 0, "feedback": feedback_context}
        )
        return feedback_context

    async def run_improvement(
        self,
        initial_question: str,
        current_responses: dict[str, str],
        improvement_instruction: str,
        improvement_context: dict[str, dict[str, str]] | None = None,
        other_responses: dict[str, str] | None = None,
    ) -> dict[str, str]:
        self.logger.info(
            "Improvement Phase", extra={"display_type": "section_header"}
        )

        # Pre-fetch knowledge bank context (async) before entering sync prompt builder
        kb_context = await self._get_knowledge_bank_context()

        def build_improvement_prompt(model_key: str, model: BaseModel) -> str:
            display_name = self.anon_mapping[model_key]
            own_answer = current_responses[display_name]

            return self.prompt_builder.build_improvement_prompt(
                initial_question=initial_question,
                own_answer=own_answer,
                improvement_instruction=improvement_instruction,
                kb_context=kb_context,
                improvement_context=improvement_context,
                other_responses=other_responses,
                model=model,
                display_name=display_name,
            )

        improved_responses = await self._execute_parallel_model_tasks(
            model_keys_to_run=self.active_model_keys,
            prompt_builder=build_improvement_prompt,
            context_for_logging="IMPROVEMENT",
        )

        self.logger.info(
            "Improved Responses", extra={"display_type": "section_header"}
        )
        for _display_name, _improved_text in improved_responses.items():
            # Response already logged immediately after generation
            pass

        return improved_responses

    def _select_largest_model_as_judge(self) -> str | None:
        if not self.active_model_keys:
            return None

        # Filter to only active models
        active_models = {k: self.models[k] for k in self.active_model_keys}
        best_model_key = select_model_by_capacity(
            active_models, include_max_tokens=True
        )

        if best_model_key is not None:
            model = self.models[best_model_key]
            self.logger.info(
                f"Selected {model.display_name} as emergency judge "
                f"(context_window={model.context_window}, max_tokens={model.max_tokens})"
            )

        return best_model_key

    async def run_cross_evaluation(
        self,
        initial_question: str,
        responses: dict[str, str],
        round_num: int,
    ) -> dict[str, str]:
        self.logger.info(
            f"Phase 3: Cross-Evaluation (Round {round_num})",
            extra={"display_type": "section_header"},
        )

        if self.judge_model_key:
            self.logger.info(
                f"Using dedicated judge model for evaluation: {self.judge_model_key}"
            )
            return await self._run_judge_evaluation_with_fallback(
                self.judge_model_key, initial_question, responses
            )
        else:
            self.logger.info(
                "Using cross-evaluation (peer review). All models will evaluate each other."
            )
            result = await self._run_peer_evaluation(
                initial_question, responses
            )

            has_valid_scores = False
            if (
                hasattr(self, "evaluation_scores")
                and self.evaluation_scores
                and isinstance(
                    next(iter(self.evaluation_scores.values()), None), dict
                )
            ):
                has_valid_scores = any(self.evaluation_scores.values())

            if not has_valid_scores:
                self.logger.warning(
                    "ALL peer evaluators failed to provide valid scores. "
                    "Falling back to JUDGE MODE with largest model."
                )

                emergency_judge = self._select_largest_model_as_judge()
                if emergency_judge:
                    self.logger.info(
                        f"EMERGENCY JUDGE MODE: Using {self.models[emergency_judge].display_name}"
                    )
                    result = await self._run_judge_evaluation(
                        emergency_judge, initial_question, responses
                    )
                else:
                    self.logger.error(
                        "CRITICAL: No model available for emergency judge fallback"
                    )

            return result

    async def _run_judge_evaluation_with_fallback(
        self,
        primary_judge_key: str,
        initial_question: str,
        collaborative_responses: dict[str, str],
    ) -> dict[str, str]:
        # Try primary judge first
        result = await self._run_judge_evaluation(
            primary_judge_key, initial_question, collaborative_responses
        )

        # If primary judge succeeded, return results
        if result and self.evaluation_scores:
            return result

        # Primary judge failed, try fallback strategies
        self.logger.warning(
            f"Primary judge {self.models[primary_judge_key].display_name} failed. "
            "Attempting fallback strategies..."
        )

        # Fallback 1: Try emergency judge (largest model)
        emergency_judge = self._select_largest_model_as_judge()
        if emergency_judge and emergency_judge != primary_judge_key:
            self.logger.info(
                f"Fallback 1: Trying emergency judge "
                f"{self.models[emergency_judge].display_name}"
            )
            result = await self._run_judge_evaluation(
                emergency_judge, initial_question, collaborative_responses
            )

            if result and self.evaluation_scores:
                self.logger.info("Emergency judge evaluation succeeded")
                return result

            self.logger.warning("Emergency judge also failed")

        # Fallback 2: Peer review as last resort
        self.logger.info(
            "Fallback 2: Falling back to peer review mode "
            "(all models evaluate each other)"
        )
        return await self._run_peer_evaluation(
            initial_question, collaborative_responses
        )

    async def _run_judge_evaluation(
        self,
        judge_model_key: str,
        initial_question: str,
        collaborative_responses: dict[str, str],
    ) -> dict[str, str]:
        judge_display_name = self.models[judge_model_key].display_name
        self.logger.info(
            f"Using {judge_display_name} as the judge.",
            extra={"display_type": "colored_text", "color": "info"},
        )

        shuffled_responses = collaborative_responses
        reverse_shuffle_mapping = {v: k for k, v in self.anon_mapping.items()}

        self.logger.info(f"Judge {judge_display_name} will evaluate:")
        for anon_name, real_name in reverse_shuffle_mapping.items():
            self.logger.info(f"  {anon_name} (actually {real_name})")

        formatted_responses = "\n\n".join(
            self.formatter.format_response_wrapper(resp_name, resp_text)
            for resp_name, resp_text in shuffled_responses.items()
        )

        code_names = list(shuffled_responses.keys())
        prompt = self.prompt_builder.build_evaluation_prompt(
            initial_question,
            formatted_responses,
            code_names,
        )

        response = await self._execute_single_model_task(
            model_key=judge_model_key,
            prompt=prompt,
            context_for_logging="JUDGE_EVAL",
        )

        if response.is_error():
            self.logger.warning(
                f"Judge {judge_display_name} evaluation failed: {response.error}"
            )
            self.evaluation_scores = {}
            self.all_evaluations = {}
            return {}

        evaluation_text = response.content
        decoded_eval = self._decode_shuffled_names(
            evaluation_text, reverse_shuffle_mapping
        )

        log_message = self.formatter.format_judge_evaluation(
            judge_display_name, decoded_eval
        )
        self.logger.info(indent_text(log_message))

        self.logger.info(
            f"\nEvaluation from Judge {judge_display_name}:",
            extra={
                "display_type": "colored_text",
                "color": judge_display_name,
            },
        )
        self.logger.info(
            decoded_eval,
            extra={
                "display_type": "colored_text",
                "color": judge_display_name,
            },
        )

        code_names = list(shuffled_responses.keys())
        raw_scores = self.score_extractor.extract_scores_from_evaluation(
            evaluation_text=evaluation_text,
            model_names=code_names,
            evaluator_name=judge_display_name,
        )

        self.all_evaluations = {judge_display_name: decoded_eval}
        self.evaluation_scores = raw_scores

        self.logger.info(
            f"Judge {judge_display_name} provided scores: {raw_scores}"
        )
        return self.all_evaluations

    async def _run_peer_evaluation(
        self,
        initial_question: str,
        collaborative_responses: dict[str, str],
    ) -> dict[str, str]:
        self.prompts.get("evaluate", "")
        all_evaluations = {}
        evaluation_scores = {}

        shuffled_responses = collaborative_responses
        reverse_shuffle_mapping = {v: k for k, v in self.anon_mapping.items()}

        self.logger.info("Peer evaluation anonymization mapping:")
        for anon_name, real_name in reverse_shuffle_mapping.items():
            self.logger.info(f"  {anon_name} = {real_name}")

        def get_responses_for_evaluator(
            evaluator_anon_name: str,
        ) -> dict[str, str]:
            self.logger.debug(
                f"Evaluator {evaluator_anon_name} will evaluate {len(shuffled_responses)} responses: {list(shuffled_responses.keys())}"
            )
            return shuffled_responses

        def build_peer_review_prompt(model_key: str, model: BaseModel) -> str:
            evaluator_anon_name = self.anon_mapping[model_key]
            evaluator_display_name = self.models[model_key].display_name

            self.logger.info(
                f"Evaluator {evaluator_display_name} (as {evaluator_anon_name}) is scoring:"
            )
            for anon_name, real_name in reverse_shuffle_mapping.items():
                self.logger.info(
                    f"  {anon_name} (labeled as {anon_name} in prompt, actually {real_name})"
                )

            responses_to_evaluate = get_responses_for_evaluator(
                evaluator_anon_name
            )
            formatted_responses = "\n\n".join(
                self.formatter.format_response_wrapper(resp_name, resp_text)
                for resp_name, resp_text in responses_to_evaluate.items()
            )
            code_names_to_evaluate = list(responses_to_evaluate.keys())
            self.logger.debug(
                f"Building evaluation prompt for {evaluator_anon_name}. "
                f"Formatted responses length: {len(formatted_responses)}, "
                f"Models to evaluate: {code_names_to_evaluate}"
            )
            prompt = self.prompt_builder.build_evaluation_prompt(
                initial_question,
                formatted_responses,
                code_names_to_evaluate,
            )
            self.logger.debug(
                f"Full evaluation prompt for {evaluator_anon_name}:\n{prompt}"
            )
            return prompt

        evaluator_responses = await self._execute_parallel_model_tasks(
            model_keys_to_run=self.active_model_keys,
            prompt_builder=build_peer_review_prompt,
            context_for_logging="PEER_EVAL",
        )

        for display_name, evaluation_text in evaluator_responses.items():
            responses_to_evaluate = get_responses_for_evaluator(display_name)
            code_names = list(responses_to_evaluate.keys())

            log_message = LOG_EVALUATOR_RESPONSE.format(
                evaluator=display_name, content=evaluation_text
            )
            self.logger.info(indent_text(log_message))
            raw_scores = self.score_extractor.extract_scores_from_evaluation(
                evaluation_text=evaluation_text,
                model_names=code_names,
                evaluator_name=display_name,
            )

            # Keep scores with anonymized keys - _get_aggregated_scores() expects them
            # raw_scores already has anonymized keys (LLM1, LLM2, etc.) from code_names
            if raw_scores:
                self.logger.info(
                    f"Extracted scores from {display_name}: {raw_scores}"
                )

            evaluation_scores[display_name] = raw_scores
            decoded_eval = self._decode_shuffled_names(
                evaluation_text, reverse_shuffle_mapping
            )

            all_evaluations[display_name] = decoded_eval
            self.logger.info(
                f"\nEvaluations from {display_name}:",
                extra={"display_type": "colored_text", "color": display_name},
            )
            self.logger.info(
                decoded_eval,
                extra={"display_type": "colored_text", "color": display_name},
            )

        self.evaluation_scores = evaluation_scores
        self.all_evaluations = all_evaluations
        return all_evaluations

    def _split_valid_invalid_evaluators(
        self,
    ) -> tuple[list[str], list[str]]:
        valid_evaluators = [
            evaluator
            for evaluator, evaluations in self.evaluation_scores.items()
            if evaluations
        ]
        invalid_evaluators = [
            evaluator
            for evaluator, evaluations in self.evaluation_scores.items()
            if not evaluations
        ]
        self.logger.info(
            f"Evaluator quality check: {len(valid_evaluators)} valid, "
            f"{len(invalid_evaluators)} invalid"
        )
        if invalid_evaluators:
            self.logger.warning(
                f"Invalid evaluators (no scores): {invalid_evaluators}"
            )
        return valid_evaluators, invalid_evaluators

    def _finalize_model_score(
        self,
        model_display_name: str,
        scores: list[float],
    ) -> float:
        if scores:
            median = statistics.median(scores)

            # Calculate score variance for transparency
            if len(scores) > 1:
                variance = statistics.variance(scores)
                stdev = statistics.stdev(scores)

                # Store variance for this model (for elimination confidence)
                if not hasattr(self, "model_score_variances"):
                    self.model_score_variances = {}
                self.model_score_variances[model_display_name] = {
                    "variance": variance,
                    "stdev": stdev,
                    "num_judges": len(scores),
                }

                self.logger.info(
                    f"  → Median score for {model_display_name}: {median:.2f} "
                    f"(stdev={stdev:.2f}, from {len(scores)} judges)"
                )
            else:
                self.logger.info(
                    f"  → Median score for {model_display_name}: {median:.2f} (from 1 judge only)"
                )

            self._display_model_score(
                model_display_name, median, "Median score"
            )
            return median
        else:
            self.logger.warning(
                f"No valid scores found for {model_display_name}. Assigning a penalty score of 0.0."
            )
            return 0.0

    def _normalize_judge_scores(
        self, judge_scores: dict[str, float], judge_name: str
    ) -> dict[str, float]:
        """Normalize scores from a single judge to account for grading bias.

        Uses z-score normalization to handle harsh/generous graders:
        - Harsh judge: gives low scores to everyone (mean=6) -> normalized up
        - Generous judge: gives high scores to everyone (mean=9) -> normalized down

        Args:
            judge_scores: Raw scores from this judge {model_name: score}
            judge_name: Name of the judge (for logging)

        Returns:
            Normalized scores {model_name: normalized_score} in [1, 10] range
        """
        self.logger.debug(
            f"Normalizing {judge_name}'s scores: {len(judge_scores)} models"
        )
        if not judge_scores or len(judge_scores) < 2:
            self.logger.debug(
                f"Skipping normalization for {judge_name}: < 2 scores"
            )
            return judge_scores

        scores_list = list(judge_scores.values())
        mean_score = statistics.mean(scores_list)

        # Calculate std dev, handle edge case where all scores are the same
        if len(set(scores_list)) == 1:
            self.logger.debug(
                f"All scores identical for {judge_name}, no normalization needed"
            )
            return judge_scores

        std_dev = statistics.stdev(scores_list)

        # Avoid division by zero
        if std_dev == 0:
            self.logger.debug(
                f"Zero std dev for {judge_name}, no normalization needed"
            )
            return judge_scores

        # Z-score normalization, then rescale to [1, 10]
        normalized = {}
        z_scores = []

        for _model_name, score in judge_scores.items():
            z_score = (score - mean_score) / std_dev
            z_scores.append(z_score)

        # Rescale z-scores to [1, 10] range
        # Typical z-scores are in [-3, +3], but we'll use actual min/max
        if z_scores:
            min_z = min(z_scores)
            max_z = max(z_scores)
            z_range = max_z - min_z

            if z_range > 0:
                for model_name, score in judge_scores.items():
                    z_score = (score - mean_score) / std_dev
                    # Map [min_z, max_z] -> [1, 10]
                    normalized_score = (
                        1.0 + ((z_score - min_z) / z_range) * 9.0
                    )
                    normalized[model_name] = normalized_score
            else:
                # All z-scores the same (shouldn't happen)
                normalized = dict.fromkeys(judge_scores.keys(), 5.5)

        # Log the normalization
        self.logger.info(
            f"Normalized {judge_name}'s scores (mean={mean_score:.2f}, std={std_dev:.2f})"
        )

        return normalized

    def _extract_numeric_scores(
        self, raw_scores: dict[str, Any], evaluator: str
    ) -> dict[str, float]:
        self.logger.debug(
            f"Extracting numeric scores from {evaluator}: {len(raw_scores)} raw scores"
        )
        numeric_scores = {}
        failed_extractions = []
        for model_name, score in raw_scores.items():
            if isinstance(score, (int, float)):
                validated = self.score_extractor.normalize_score(
                    float(score), evaluator
                )
                if validated is not None:
                    numeric_scores[model_name] = validated
                    self.logger.debug(f"{model_name}: {score} → {validated}")
                else:
                    failed_extractions.append(f"{model_name}={score}")
            else:
                failed_extractions.append(
                    f"{model_name}={type(score).__name__}"
                )
        if failed_extractions:
            self.logger.debug(
                f"Failed to extract scores: {', '.join(failed_extractions)}"
            )
        self.logger.info(
            f"Extracted {len(numeric_scores)}/{len(raw_scores)} scores from {evaluator}"
        )
        return numeric_scores

    def _detect_self_scoring_bias(
        self, numeric_scores: dict[str, float], evaluator: str
    ) -> float | None:
        self.logger.debug(f"Detecting self-scoring bias for {evaluator}")
        if evaluator not in numeric_scores:
            self.logger.debug(
                f"{evaluator} not in scores, skipping bias detection"
            )
            return None

        self_score = numeric_scores[evaluator]
        other_scores = [s for m, s in numeric_scores.items() if m != evaluator]

        if not other_scores:
            self.logger.debug(f"No other scores to compare for {evaluator}")
            return None

        avg_others = statistics.mean(other_scores)
        bias = self_score - avg_others

        if bias > 1.5:
            self.logger.warning(
                f"Potential self-scoring bias detected: {evaluator} gave itself "
                f"{self_score:.1f} vs {avg_others:.1f} avg to others (bias: +{bias:.1f})"
            )
        elif bias < -1.5:
            self.logger.info(
                f"{evaluator} scored itself lower than others: "
                f"{self_score:.1f} vs {avg_others:.1f} avg (bias: {bias:.1f})"
            )
        else:
            self.logger.debug(
                f"{evaluator} self-score: {self_score:.1f}, others avg: {avg_others:.1f} (bias: {bias:+.1f})"
            )

        return bias

    def _normalize_evaluator_scores(
        self, evaluator: str
    ) -> tuple[float | None, dict[str, float]] | None:
        self.logger.debug(f"Normalizing scores from evaluator: {evaluator}")
        raw_scores = self.evaluation_scores[evaluator]
        if not isinstance(raw_scores, dict):
            self.logger.warning(
                f"{evaluator} has non-dict scores, skipping normalization"
            )
            return None

        numeric_scores = self._extract_numeric_scores(raw_scores, evaluator)
        if not numeric_scores:
            self.logger.warning(
                f"No numeric scores extracted from {evaluator}"
            )
            return None

        bias = self._detect_self_scoring_bias(numeric_scores, evaluator)
        normalized = self._normalize_judge_scores(numeric_scores, evaluator)
        if bias is not None:
            self.logger.info(
                f"{evaluator} normalization complete with bias={bias:+.2f}"
            )
            return bias, normalized

        self.logger.info(
            f"{evaluator} normalization complete, no bias detected"
        )
        return None, normalized

    def _log_self_scoring_bias_summary(
        self, self_scoring_biases: dict[str, float]
    ) -> None:
        self.self_scoring_biases = self_scoring_biases
        self.logger.info(
            f"Self-scoring bias summary: {len(self_scoring_biases)} models scored themselves"
        )
        for model, bias in sorted(
            self_scoring_biases.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            bias_str = f"{bias:+.2f}"
            if abs(bias) > 1.5:
                self.logger.info(f"   • {model}: {bias_str} (significant)")
            else:
                self.logger.debug(f"   • {model}: {bias_str}")

    def _collect_scores_for_model(
        self,
        model_display_name: str,
        normalized_evaluation_scores: dict[str, dict[str, float]],
        valid_evaluators: list[str],
    ) -> list[float]:
        scores: list[float] = []
        self.logger.info(f"Collecting scores for {model_display_name}:")

        for evaluator in valid_evaluators:
            if evaluator not in normalized_evaluation_scores:
                continue

            normalized_scores = normalized_evaluation_scores[evaluator]
            if model_display_name not in normalized_scores:
                continue

            norm_score = normalized_scores[model_display_name]
            scores.append(norm_score)

            is_self_score = evaluator == model_display_name
            score_type = " (self-score)" if is_self_score else ""
            self.logger.info(
                f"  - {evaluator} gave {model_display_name} a normalized score of {norm_score:.2f}{score_type}"
            )

        return scores

    def _aggregate_peer_review_scores(self) -> dict[str, float]:
        self.logger.info("Aggregating scores from peer-review format.")
        aggregated_scores: dict[str, float] = {}

        valid_evaluators, invalid_evaluators = (
            self._split_valid_invalid_evaluators()
        )

        if invalid_evaluators:
            self.logger.warning(
                f"{len(invalid_evaluators)} evaluator(s) provided invalid evaluations and were excluded: {', '.join(invalid_evaluators)}"
            )

        normalized_evaluation_scores = {}
        self_scoring_biases = {}

        for evaluator in valid_evaluators:
            result = self._normalize_evaluator_scores(evaluator)
            if result is None:
                continue

            bias, normalized_scores = result
            if bias is not None:
                self_scoring_biases[evaluator] = bias

            if normalized_scores:
                normalized_evaluation_scores[evaluator] = normalized_scores

        if self_scoring_biases:
            self._log_self_scoring_bias_summary(self_scoring_biases)

        for model_display_name in [
            self.anon_mapping[k] for k in self.active_model_keys
        ]:
            scores = self._collect_scores_for_model(
                model_display_name,
                normalized_evaluation_scores,
                valid_evaluators,
            )
            aggregated_scores[model_display_name] = self._finalize_model_score(
                model_display_name, scores
            )

        return aggregated_scores

    def _aggregate_judge_scores(self) -> dict[str, float]:
        self.logger.info("Processing scores from single-judge format.")
        result_scores: dict[str, float] = {}
        evaluator_name = next(iter(self.all_evaluations.keys()), "judge")

        for model_name, score in self.evaluation_scores.items():
            if isinstance(score, (int, float)):
                score_float = float(score)
                normalized_score = self.score_extractor.normalize_score(
                    score_float, evaluator_name
                )
                if normalized_score is not None:
                    self._display_model_score(
                        model_name, normalized_score, "Judge score"
                    )
                    result_scores[model_name] = normalized_score
                else:
                    self.logger.warning(
                        f"Judge gave {model_name} an invalid score of {score_float} (rejected)"
                    )

        return result_scores

    def _get_aggregated_scores(self) -> dict[str, float]:
        first_score_value = next(iter(self.evaluation_scores.values()))
        is_peer_review = isinstance(first_score_value, dict)

        if is_peer_review:
            return self._aggregate_peer_review_scores()
        else:
            return self._aggregate_judge_scores()

    def _handle_no_evaluation_scores(self) -> tuple[str, str]:
        self.logger.error(
            "No evaluation scores available for elimination decision. Falling back to random selection."
        )
        active_names = [self.anon_mapping[k] for k in self.active_model_keys]
        chosen = self.anonymizer.rng.choice(active_names)
        self.elimination_reason = (
            "Random selection (no evaluation scores available)"
        )
        self.elimination_score = None
        return chosen, self.anonymizer.rng.choice(active_names)

    def _select_leader_from_scored_models(
        self, active_scores: dict[str, float]
    ) -> str:
        self.logger.debug(
            f"Selecting leader from {len(active_scores)} scored models"
        )
        max_score = max(active_scores.values())
        models_with_max = [
            name for name, score in active_scores.items() if score == max_score
        ]
        leader = self.anonymizer.rng.choice(models_with_max)
        if len(models_with_max) > 1:
            self.logger.info(
                f"Tie for leader (score={max_score:.2f}): {models_with_max}, "
                f"randomly selected {leader}"
            )
        else:
            self.logger.info(
                f"Leader selected: {leader} (score={max_score:.2f})"
            )
        return leader

    def _handle_unscored_models(
        self,
        unscored_models: set[str],
        active_scores: dict[str, float],
        all_active_models: set[str],
    ) -> tuple[str, str]:
        self.logger.info(f"Handling {len(unscored_models)} unscored models")
        lowest_model_name = self.anonymizer.rng.choice(list(unscored_models))
        self.logger.warning(
            f"Models {list(unscored_models)} were not scored by any evaluator. "
            f"Randomly selecting {lowest_model_name} for elimination."
        )
        self.elimination_reason = (
            "Random selection (model was not scored by any evaluator)"
        )
        self.elimination_score = None

        if not active_scores:
            # All models were unscored
            remaining_models = list(all_active_models - {lowest_model_name})
            highest_model_name = (
                self.anonymizer.rng.choice(remaining_models)
                if remaining_models
                else lowest_model_name
            )
            self.logger.warning(
                "No models received scores. Randomly selecting leader."
            )
        else:
            # At least one model was scored
            highest_model_name = self._select_leader_from_scored_models(
                active_scores
            )

        return lowest_model_name, highest_model_name

    def _handle_critical_no_scores(
        self, all_active_models: set[str]
    ) -> tuple[str, str]:
        self.logger.error(
            "CRITICAL: No models have scores and no unscored models found. Cannot determine ranking."
        )
        active_names = list(all_active_models)
        chosen = self.anonymizer.rng.choice(active_names)
        self.elimination_reason = (
            "Random selection (critical error - no valid scores)"
        )
        self.elimination_score = None
        return chosen, self.anonymizer.rng.choice(active_names)

    def _select_lowest_ranked_model(
        self, active_scores: dict[str, float]
    ) -> tuple[str, float]:
        self.logger.debug(
            f"Selecting lowest-ranked model from {len(active_scores)} scores"
        )
        min_score = min(active_scores.values())
        models_with_min = [
            name for name, score in active_scores.items() if score == min_score
        ]

        if len(models_with_min) > 1:
            lowest_model_name = self.anonymizer.rng.choice(models_with_min)
            self.logger.warning(
                f"Tie for lowest score ({min_score:.2f}): {models_with_min}. "
                f"Randomly selected {lowest_model_name} for elimination."
            )
            self.elimination_reason = f"Random selection among tied models (tied at score {min_score:.2f})"
        else:
            lowest_model_name = models_with_min[0]
            self.logger.info(
                f"Eliminating {lowest_model_name} (lowest score: {min_score:.2f})"
            )
            self.elimination_reason = "Lowest score in evaluation"

        self.elimination_score = min_score
        return lowest_model_name, min_score

    def _select_highest_ranked_model(
        self, active_scores: dict[str, float]
    ) -> str:
        self.logger.debug(
            f"Selecting highest-ranked model from {len(active_scores)} scores"
        )
        max_score = max(active_scores.values())
        models_with_max = [
            name for name, score in active_scores.items() if score == max_score
        ]

        if len(models_with_max) > 1:
            highest_model_name = self.anonymizer.rng.choice(models_with_max)
            self.logger.info(
                f"Tie for highest score ({max_score:.2f}): {models_with_max}. Randomly selected {highest_model_name} as leader."
            )
        else:
            highest_model_name = models_with_max[0]
            self.logger.info(
                f"Leader: {highest_model_name} (highest score: {max_score:.2f})"
            )

        return highest_model_name

    def _resolve_same_model_tie(
        self,
        lowest_model_name: str,
        highest_model_name: str,
        active_scores: dict[str, float],
        min_score: float,
    ) -> str:
        self.logger.debug("Checking for same-model tie resolution")
        if lowest_model_name == highest_model_name and len(active_scores) > 1:
            self.logger.warning(
                f"All models tied with score {min_score:.2f}. Randomly selecting for elimination and leadership."
            )
            other_models = [
                name
                for name in active_scores.keys()
                if name != lowest_model_name
            ]
            if other_models:
                highest_model_name = self.anonymizer.rng.choice(other_models)
            self.logger.info(
                f"Randomly selecting {lowest_model_name} to be eliminated and {highest_model_name} to lead."
            )
            self.elimination_reason = (
                f"Random selection (all models tied at {min_score:.2f})"
            )
            self.elimination_score = min_score

        return highest_model_name

    def _handle_all_models_scored(
        self, active_scores: dict[str, float]
    ) -> tuple[str, str]:
        self.logger.info(
            "All models received scores. Determining winner and loser based on scores."
        )

        lowest_model_name, min_score = self._select_lowest_ranked_model(
            active_scores
        )
        highest_model_name = self._select_highest_ranked_model(active_scores)
        highest_model_name = self._resolve_same_model_tie(
            lowest_model_name, highest_model_name, active_scores, min_score
        )

        return lowest_model_name, highest_model_name

    def _finalize_ranking_results(
        self,
        lowest_model_name: str,
        highest_model_name: str,
        active_scores: dict[str, float],
    ) -> tuple[str, str]:
        self.current_leader_key = next(
            (
                key
                for key, name in self.anon_mapping.items()
                if name == highest_model_name
            ),
            None,
        )

        highest_score_val = active_scores.get(highest_model_name, float("nan"))
        lowest_score_val = active_scores.get(lowest_model_name, 0.0)

        self.logger.info(
            f"\nHighest-ranked model: {highest_model_name} with score {highest_score_val:.2f}/10",
            extra={"display_type": "colored_text", "color": "success"},
        )
        self.logger.info(
            f"Lowest-ranked model: {lowest_model_name} with score {lowest_score_val:.2f}/10",
            extra={"display_type": "colored_text", "color": "warning"},
        )

        return lowest_model_name, highest_model_name

    def determine_lowest_and_highest_ranked_models(
        self,
    ) -> tuple[str, str]:
        if (
            not hasattr(self, "evaluation_scores")
            or not self.evaluation_scores
        ):
            return self._handle_no_evaluation_scores()

        self.logger.info(
            "Aggregating Evaluation Scores",
            extra={"display_type": "section_header"},
        )
        aggregated_scores = self._get_aggregated_scores()

        all_active_models = {
            self.anon_mapping[k] for k in self.active_model_keys
        }
        active_scores = {
            k: v
            for k, v in aggregated_scores.items()
            if k in all_active_models and v is not None
        }
        unscored_models = all_active_models - set(active_scores.keys())

        if unscored_models:
            lowest, highest = self._handle_unscored_models(
                unscored_models, active_scores, all_active_models
            )
        elif not active_scores:
            return self._handle_critical_no_scores(all_active_models)
        else:
            lowest, highest = self._handle_all_models_scored(active_scores)

        return self._finalize_ranking_results(lowest, highest, active_scores)

    def _get_phase_2_criticisms(self) -> Any | None:
        if not self.criticism_history:
            return None
        for crit_entry in self.criticism_history:
            if crit_entry.get("round") == 0:
                return crit_entry.get("criticisms")
        return None

    def _get_phase_2_feedback(self) -> Any | None:
        if not self.feedback_history:
            return None
        for feedback_entry in self.feedback_history:
            if feedback_entry.get("round") == 0:
                return feedback_entry.get("feedback")
        return None

    def _build_phase_2_data(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        phase_2_criticisms = self._get_phase_2_criticisms()
        phase_2_feedback = self._get_phase_2_feedback()

        phase_2_data: dict[str, Any] = {}

        # Always include the improved answers - this is the key data
        improved_answers = all_previous_answers[1].copy()

        if phase_2_criticisms:
            phase_2_data["criticisms"] = phase_2_criticisms
            phase_2_data["improved_answers"] = improved_answers
            return {
                "Phase 2: Cross-Criticism & Self-Improvement": phase_2_data
            }
        elif phase_2_feedback:
            phase_2_data["feedback"] = phase_2_feedback
            phase_2_data["enhanced_answers"] = improved_answers
            return {
                "Phase 2: Positive Reinforcement & Strength Amplification": phase_2_data
            }
        else:
            # Collaborative mode - just the improved answers, no extra fields
            return {"Phase 2: Collaborative Analysis": improved_answers}

    def _add_round_evaluations(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        if hasattr(self, "evaluation_history") and elimination_round - 1 < len(
            self.evaluation_history
        ):
            eval_data = self.evaluation_history[elimination_round - 1]
            round_data["evaluations"] = eval_data.get("evaluations", {})
            round_data["scores"] = eval_data.get("scores", {})

    def _add_round_criticisms(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        if not hasattr(self, "criticism_history"):
            return
        for crit_entry in self.criticism_history:
            if crit_entry.get("round") == elimination_round:
                round_data["criticisms"] = crit_entry.get("criticisms", {})
                break

    def _add_round_feedback(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        if not hasattr(self, "feedback_history"):
            return
        for feedback_entry in self.feedback_history:
            if feedback_entry.get("round") == elimination_round:
                round_data["feedback"] = feedback_entry.get("feedback", {})
                break

    def _build_elimination_rounds(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        tournament_data = {}
        for elimination_round, i in enumerate(
            range(2, len(all_previous_answers)), start=1
        ):
            round_data: dict[str, Any] = {}
            self._add_round_evaluations(round_data, elimination_round)
            self._add_round_criticisms(round_data, elimination_round)
            self._add_round_feedback(round_data, elimination_round)
            round_data["refined_answers"] = all_previous_answers[i].copy()
            tournament_data[f"Elimination Round {elimination_round}"] = (
                round_data
            )
        return tournament_data

    def _build_tournament_history(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        tournament_history: dict[str, Any] = {}

        if len(all_previous_answers) > 0:
            tournament_history["Phase 1: Initial Answers"] = (
                all_previous_answers[0].copy()
            )

        if len(all_previous_answers) > 1:
            tournament_history.update(
                self._build_phase_2_data(all_previous_answers)
            )

        tournament_history.update(
            self._build_elimination_rounds(all_previous_answers)
        )

        return tournament_history

    async def _save_champion_report(
        self,
        initial_question: str,
        final_model_anon: str,
        champion_answer: str,
        all_previous_answers: list[dict[str, str]],
    ) -> None:
        from arbitrium_core.domain.reporting.provenance import ProvenanceReport

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tournament_history = self._build_tournament_history(
            all_previous_answers
        )

        champion_model = f"{final_model_anon} (model: {self.models[self.active_model_keys[0]].full_display_name})"

        tournament_data = {
            **self.cost_tracker.get_summary(),
            "eliminated_models": self.eliminated_models.copy(),
            "complete_tournament_history": tournament_history,
        }

        provenance = ProvenanceReport(
            question=initial_question,
            champion_model=champion_model,
            champion_answer=champion_answer,
            tournament_data=tournament_data,
        )

        outputs_dir = str(self.host.base_dir)
        saved_paths = await provenance.save_to_file(outputs_dir, timestamp)

        for file_type, file_path in saved_paths.items():
            self.logger.info(f"{file_type}: {file_path}")

    def _display_cost_summary(self) -> None:
        self.cost_tracker.display_summary()

    async def run(self, initial_question: str) -> str:
        try:
            return await self.runner.run(initial_question)
        finally:
            self._display_cost_summary()
            self.logger.info(
                "Arbitrium Framework Tournament Complete",
                extra={"display_type": "section_header"},
            )
            # self.display.reset() # This needs to be handled by the event handler

import asyncio
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from certamen_core.domain.interrogation.interrogator import (
        AdversarialInterrogator,
    )

from certamen_core.domain.knowledge.bank import EnhancedKnowledgeBank
from certamen_core.domain.model_selection import select_model_by_capacity
from certamen_core.domain.prompts import (
    LOG_EVALUATOR_RESPONSE,
    PromptBuilder,
    PromptFormatter,
)
from certamen_core.domain.tournament.aggregator import ScoreAggregator
from certamen_core.domain.tournament.anonymizer import ModelAnonymizer
from certamen_core.domain.tournament.budget import CostTracker
from certamen_core.domain.tournament.history import TournamentHistoryBuilder
from certamen_core.domain.tournament.ranking import RankingEngine
from certamen_core.domain.tournament.report import ReportGenerator
from certamen_core.domain.tournament.scoring import ScoreExtractor
from certamen_core.ports.llm import BaseModel, ModelResponse
from certamen_core.ports.similarity import SimilarityEngine
from certamen_core.ports.tournament import EventHandler, HostEnvironment
from certamen_core.shared.constants import (
    PLACEHOLDER_RESPONSES,
    VARIANCE_HIGH_CONFIDENCE_THRESHOLD,
    VARIANCE_MEDIUM_CONFIDENCE_THRESHOLD,
)
from certamen_core.shared.logging import get_contextual_logger
from certamen_core.shared.text import indent_text, strip_meta_commentary

_DEFAULT_DB_PATH = "certamen_knowledge.db"


class TournamentRunner:
    def __init__(self, comparison_instance: "ModelComparison") -> None:
        self.comp = comparison_instance
        self.event_handler = comparison_instance.event_handler
        self.logger = comparison_instance.logger

    async def _run_tournament_phases(self, initial_question: str) -> str:
        if not await self._run_initial_phase(initial_question):
            return "No valid initial responses. Tournament cannot proceed."
        await self._run_interrogation_phase(
            initial_question, self.comp.previous_answers[0]
        )
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
        self.comp.interrogation_context = ""
        self.comp.model_score_variances = {}
        self.comp.self_scoring_biases = {}
        self.comp.elimination_reason = ""
        self.comp.elimination_score = None
        self._knowledge_map: Any = None
        self._disagreement_reports: list[Any] = []

        try:
            await self._load_prior_knowledge(initial_question)
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
            "PHASE 1 COMPLETE: Got %s initial responses",
            len(initial_responses),
        )
        return True

    def _build_phase_skip_message(
        self, phase_name: str, round_num: int | None
    ) -> str:
        if round_num is not None:
            return f"{phase_name} is disabled. Skipping round {round_num}."
        return f"{phase_name} is disabled. Skipping."

    def _log_phase_header(
        self, phase_name: str, round_num: int | None
    ) -> None:
        if round_num is None:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("%s", phase_name)
            self.logger.info("=" * 80)
        else:
            self.logger.info("\nROUND %s: %s", round_num, phase_name)

    async def _collect_feedback_context(
        self,
        initial_question: str,
        source_answers: dict[str, str],
        phase_config: dict[str, Any],
        round_num: int | None,
    ) -> dict[str, dict[str, str]] | None:
        if not phase_config.get("feedback_enabled", False):
            return None
        self.logger.info("Collecting feedback from models...")
        feedback_context = await self.comp.run_feedback(
            initial_question,
            source_answers,
            feedback_instruction=phase_config.get(
                "feedback_instruction", "Provide feedback for this answer."
            ),
            round_number=round_num if round_num is not None else 0,
        )
        if not feedback_context:
            self.logger.warning("No feedback collected, proceeding without it")
            return None
        return feedback_context

    def _merge_evaluation_into_feedback(
        self,
        feedback_context: dict[str, dict[str, str]] | None,
        evaluation_context: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        if feedback_context is None:
            return evaluation_context
        for model_name, eval_feedbacks in evaluation_context.items():
            if model_name not in feedback_context:
                feedback_context[model_name] = {}
            feedback_context[model_name].update(eval_feedbacks)
        return feedback_context

    def _log_improvement_completion(
        self,
        phase_name: str,
        round_num: int | None,
        response_count: int,
        action_word: str,
    ) -> None:
        if round_num is None:
            self.logger.info(
                "%s COMPLETE: Got %s %s responses",
                phase_name,
                response_count,
                action_word,
            )
        else:
            self.logger.info(
                "ROUND %s COMPLETE: Got %s %s responses",
                round_num,
                response_count,
                action_word,
            )

    async def _run_improvement_phase(
        self,
        initial_question: str,
        config_key: str,
        phase_name: str,
        answer_index: int,
        round_num: int | None = None,
        evaluation_context: dict[str, dict[str, str]] | None = None,
    ) -> bool:
        phase_config = self.comp.config.get(config_key, {})

        if not phase_config.get("enabled", True):
            self.logger.info(
                self._build_phase_skip_message(phase_name, round_num)
            )
            return True

        self._log_phase_header(phase_name, round_num)

        source_answers = self.comp.previous_answers[answer_index]

        feedback_context = await self._collect_feedback_context(
            initial_question, source_answers, phase_config, round_num
        )

        if evaluation_context:
            feedback_context = self._merge_evaluation_into_feedback(
                feedback_context, evaluation_context
            )

        action_word = "refined" if round_num is not None else "improved"
        self.logger.info("Generating %s responses...", action_word)

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
        self._log_improvement_completion(
            phase_name, round_num, len(responses), action_word
        )
        return True

    def _resolve_model_key_by_anon_name(self, anon_name: str) -> str | None:
        return next(
            (k for k, v in self.comp.anon_mapping.items() if v == anon_name),
            None,
        )

    async def _interrogate_pair(
        self,
        interrogator: "AdversarialInterrogator",
        examiner_name: str,
        target_name: str,
        responses: dict[str, str],
        initial_question: str,
        max_q: int,
    ) -> list[str]:
        examiner_key = self._resolve_model_key_by_anon_name(examiner_name)
        target_key = self._resolve_model_key_by_anon_name(target_name)
        if not examiner_key or not target_key:
            return []
        examiner_model = self.comp.models.get(examiner_key)
        target_model = self.comp.models.get(target_key)
        if not examiner_model or not target_model:
            return []
        questions = await interrogator.generate_questions(
            examiner_model=examiner_model,
            target_response=responses[target_name],
            other_response=responses[examiner_name],
            question=initial_question,
            max_questions=max_q,
        )
        qa = await interrogator.conduct_interrogation(
            target_model=target_model,
            questions=questions,
            question=initial_question,
            own_response=responses[target_name],
        )
        return [f"Q: {q}\nA: {a}" for q, a in qa.items()]

    async def _run_interrogation_round(
        self,
        interrogator: "AdversarialInterrogator",
        responses: dict[str, str],
        model_names: list[str],
        initial_question: str,
        max_q: int,
    ) -> list[str]:
        tasks = [
            self._interrogate_pair(
                interrogator,
                examiner,
                target,
                responses,
                initial_question,
                max_q,
            )
            for i, examiner in enumerate(model_names)
            for j, target in enumerate(model_names)
            if i != j
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            item
            for result in results
            if isinstance(result, list)
            for item in result
        ]

    async def _extract_interrogation_insights(
        self,
        qa_pairs: list[str],
        extractor_key: str,
        phase_tag: str,
    ) -> list[str]:
        from certamen_core.shared.constants import INTERROGATION_INSIGHT_PROMPT
        from certamen_core.shared.text import parse_insight_lines

        prompt = INTERROGATION_INSIGHT_PROMPT.format(
            text="\n\n".join(qa_pairs)
        )
        response = await self.comp.execute_single_model_task(
            model_key=extractor_key,
            prompt=prompt,
            context_for_logging=phase_tag,
        )
        if response.is_error():
            self.logger.warning(
                "Interrogation insight extraction failed: %s", response.error
            )
            return []
        insights = parse_insight_lines(
            response.content, min_length=10, skip_apologies=True
        )
        if insights:
            await self.comp.knowledge_bank._add_insights_to_db(
                insights, phase_tag.lower(), 0
            )
        return insights

    async def _run_interrogation_round2(
        self,
        interrogator: "AdversarialInterrogator",
        current_responses: dict[str, str],
        model_names: list[str],
        initial_question: str,
        max_q: int,
        extractor_key: str,
        round1_insights: list[str],
    ) -> list[str]:
        round2_context = "\n".join(f"- {i}" for i in round1_insights[:10])
        round2_responses = {
            n: f"{t}\n\n[Cross-examination findings:]\n{round2_context}"
            for n, t in current_responses.items()
        }
        r2_qa = await self._run_interrogation_round(
            interrogator,
            round2_responses,
            model_names,
            initial_question,
            max_q,
        )
        if not r2_qa:
            return []
        return await self._extract_interrogation_insights(
            r2_qa, extractor_key, "INTERROGATION_R2_EXTRACTION"
        )

    async def _run_interrogation_phase(
        self,
        initial_question: str,
        current_responses: dict[str, str],
    ) -> None:
        if not self.comp.features.get("interrogation_enabled", True):
            return
        if len(current_responses) < 2:
            return

        from certamen_core.domain.interrogation.interrogator import (
            AdversarialInterrogator,
        )

        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            "INTERROGATION PHASE: Cross-examining models for hidden knowledge"
        )
        self.logger.info("=" * 80)

        interrogator = AdversarialInterrogator(self.comp.semaphore)
        max_q = int(self.comp.features.get("interrogation_max_questions", 4))
        interrogation_rounds = int(
            self.comp.features.get("interrogation_rounds", 1)
        )
        model_names = list(current_responses.keys())

        all_qa = await self._run_interrogation_round(
            interrogator,
            current_responses,
            model_names,
            initial_question,
            max_q,
        )

        if not all_qa:
            self.logger.info("INTERROGATION COMPLETE: No Q&A pairs extracted")
            return

        self.comp.interrogation_context = "\n\n".join(all_qa)

        extractor_key = self.comp.judge_model_key or (
            self.comp.active_model_keys[0]
            if self.comp.active_model_keys
            else None
        )
        if not extractor_key or extractor_key not in self.comp.models:
            self.logger.warning(
                "No model available for interrogation insight extraction"
            )
            return

        insights = await self._extract_interrogation_insights(
            all_qa, extractor_key, "INTERROGATION_EXTRACTION"
        )

        if interrogation_rounds >= 2 and insights:
            r2_insights = await self._run_interrogation_round2(
                interrogator,
                current_responses,
                model_names,
                initial_question,
                max_q,
                extractor_key,
                insights,
            )
            insights.extend(r2_insights)

        self.logger.info(
            "INTERROGATION COMPLETE: %s Q&A pairs → %s insights extracted",
            len(all_qa),
            len(insights),
        )

    async def _run_disagreement_phase(
        self,
        initial_question: str,
        responses: dict[str, str],
    ) -> list[object]:
        if not self.comp.features.get(
            "disagreement_investigation_enabled", True
        ):
            return []
        if len(responses) < 2:
            return []

        from certamen_core.domain.disagreement.detector import (
            DisagreementDetector,
        )
        from certamen_core.domain.disagreement.resolver import (
            DisagreementInvestigator,
        )

        judge_key = self.comp.judge_model_key or (
            self.comp.active_model_keys[0]
            if self.comp.active_model_keys
            else None
        )
        if not judge_key:
            return []
        judge_model = self.comp.models.get(judge_key)
        if not judge_model:
            return []

        # Map anonymized names back to display names for better readability
        real_responses: dict[str, str] = {}
        for anon_name, text in responses.items():
            real_key = next(
                (
                    k
                    for k, v in self.comp.anon_mapping.items()
                    if v == anon_name
                ),
                None,
            )
            if real_key:
                model = self.comp.models[real_key]
                display = model.display_name or real_key
                real_responses[display] = text

        detector = DisagreementDetector()
        disagreements = await detector.detect_disagreements(
            real_responses, judge_model, initial_question
        )

        if not disagreements:
            return []

        self.logger.info(
            "DISAGREEMENT PHASE: Found %s disagreements, investigating...",
            len(disagreements),
        )

        display_to_model: dict[str, BaseModel] = {}
        for key, model in self.comp.models.items():
            if (
                key in self.comp.active_model_keys
                or key == self.comp.judge_model_key
            ):
                display = model.display_name or key
                display_to_model[display] = model

        investigator = DisagreementInvestigator()
        reports = await asyncio.gather(
            *[
                investigator.investigate(d, display_to_model, initial_question)
                for d in disagreements
            ],
            return_exceptions=True,
        )

        valid: list[Any] = [
            r for r in reports if not isinstance(r, BaseException)
        ]
        self.logger.info(
            "DISAGREEMENT PHASE COMPLETE: Investigated %s disagreements",
            len(valid),
        )
        return valid

    async def _build_knowledge_map(
        self,
        initial_question: str,
        synthesis: str,
        champion_model_key: str,
    ) -> Any:
        if not self.comp.features.get("knowledge_map_enabled", True):
            return None

        from certamen_core.domain.knowledge_map.builder import (
            KnowledgeMapBuilder,
        )

        judge_key = self.comp.judge_model_key or champion_model_key
        judge_model = self.comp.models.get(judge_key)
        if not judge_model:
            return None

        all_responses = self._collect_all_final_responses()
        builder = KnowledgeMapBuilder()

        try:
            km = await builder.build(
                question=initial_question,
                all_responses=all_responses,
                synthesis=synthesis,
                champion_model=champion_model_key,
                judge_model=judge_model,
                disagreements=self._disagreement_reports,
            )
            km.exploration_branches = (
                await builder.generate_exploration_branches(km, judge_model)
            )
            if self.comp.features.get("persistence_enabled", True):
                try:
                    from certamen_core.infrastructure.persistence.knowledge_store import (
                        PersistentKnowledgeStore,
                    )

                    db_path = self.comp.features.get(
                        "persistence_db_path", _DEFAULT_DB_PATH
                    )
                    store = PersistentKnowledgeStore(db_path)
                    known = await store.get_all_branch_questions()
                    km.exploration_branches = [
                        b for b in km.exploration_branches if b not in known
                    ]
                except Exception as e:
                    self.logger.debug("Branch dedup failed: %s", e)
            self.logger.info(
                "KNOWLEDGE MAP: Built with %s consensus items, %s disagreements, "
                "%s exploration branches",
                len(km.consensus),
                len(km.disagreements),
                len(km.exploration_branches),
            )
            return km
        except Exception as e:
            self.logger.warning("Knowledge map construction failed: %s", e)
            return None

    async def _persist_knowledge_map(self, km: Any) -> None:
        if not self.comp.features.get("persistence_enabled", True):
            return

        from certamen_core.infrastructure.persistence.knowledge_store import (
            PersistentKnowledgeStore,
        )

        db_path = self.comp.features.get(
            "persistence_db_path", _DEFAULT_DB_PATH
        )
        store = PersistentKnowledgeStore(db_path)
        tournament_id = getattr(self, "_tournament_id", "unknown")
        try:
            await store.store_knowledge_map(km, tournament_id)
            self.logger.info("Knowledge map persisted to %s", db_path)
        except Exception as e:
            self.logger.warning("Failed to persist knowledge map: %s", e)

    async def _load_prior_knowledge(self, initial_question: str) -> None:
        if not self.comp.features.get("persistence_enabled", True):
            return
        try:
            from certamen_core.infrastructure.persistence.knowledge_store import (
                PersistentKnowledgeStore,
            )

            db_path = self.comp.features.get(
                "persistence_db_path", _DEFAULT_DB_PATH
            )
            store = PersistentKnowledgeStore(db_path)
            prior_claims = await store.get_relevant_prior_knowledge(
                initial_question, limit=15
            )
            if prior_claims:
                await self.comp.knowledge_bank._add_insights_to_db(
                    prior_claims, "prior_tournament", 0
                )
                self.logger.info(
                    "Loaded %s prior knowledge claims from previous tournaments",
                    len(prior_claims),
                )
        except Exception as e:
            self.logger.warning("Failed to load prior knowledge: %s", e)

    async def _run_phase_2(self, initial_question: str) -> bool:
        return await self._run_improvement_phase(
            initial_question,
            config_key="improvement_phase",
            phase_name="PHASE 2: Improvement Phase",
            answer_index=0,
        )

    async def _store_disagreement_insights(
        self, disagreement_reports: list[object]
    ) -> None:
        from certamen_core.domain.disagreement.resolver import (
            DisagreementReport,
        )

        disagreement_insights = [
            f"DISAGREEMENT [{r.resolution_status}]: {r.topic}"  # type: ignore[union-attr]
            f" — {r.neutral_analysis[:200]}"  # type: ignore[union-attr]
            for r in disagreement_reports
            if isinstance(r, DisagreementReport) and r.neutral_analysis
        ]
        if disagreement_insights:
            await self.comp.knowledge_bank._add_insights_to_db(
                disagreement_insights, "disagreement_investigation", 0
            )

    def _log_empty_evaluations_outcome(self, round_num: int) -> None:
        active_count = len(self.comp.active_model_keys)
        if active_count == 0:
            self.logger.error(
                "No evaluations in round %s and no active models remain. Tournament failed.",
                round_num,
            )
        elif active_count == 1:
            self.logger.warning(
                "No evaluations in round %s, but 1 model remains. Declaring champion.",
                round_num,
            )
        else:
            self.logger.warning(
                "No evaluations in round %s, but %s models remain. "
                "This indicates an evaluation system failure. Declaring current leader as champion.",
                round_num,
                active_count,
            )

    async def _run_elimination_rounds(self, initial_question: str) -> None:
        round_num = 1
        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            "Starting elimination rounds with %s models",
            len(self.comp.active_model_keys),
        )
        self.logger.info("=" * 80)

        while len(self.comp.active_model_keys) > 1:
            self.logger.info("\n" + "-" * 80)
            self.logger.info("ROUND %s: Cross-Evaluation Phase", round_num)
            self.logger.info("-" * 80)

            disagreement_reports = await self._run_disagreement_phase(
                initial_question, self.comp.previous_answers[-1]
            )
            self._disagreement_reports.extend(disagreement_reports)

            if disagreement_reports:
                await self._store_disagreement_insights(disagreement_reports)

            evaluations = await self.comp.run_cross_evaluation(
                initial_question, self.comp.previous_answers[-1], round_num
            )
            if not evaluations:
                self._log_empty_evaluations_outcome(round_num)
                break

            self.comp.evaluation_history.append(
                {
                    "round": round_num,
                    "evaluations": evaluations.copy(),
                    "scores": self.comp.evaluation_scores.copy(),
                }
            )

            (
                eliminated_model,
                _leader_model,
            ) = (
                self.comp.ranking_engine.determine_lowest_and_highest_ranked_models()
            )
            if not eliminated_model:
                self.logger.warning(
                    "Could not determine model to eliminate. Ending tournament."
                )
                break

            await self._handle_elimination(eliminated_model, round_num)

            if len(self.comp.active_model_keys) <= 1:
                break

            evaluation_context = self._format_evaluations_as_context()

            if not await self._run_refinement_strategy(
                initial_question,
                round_num,
                evaluation_context=evaluation_context,
            ):
                break

            round_num += 1

    def _format_evaluations_as_context(
        self,
    ) -> dict[str, dict[str, str]] | None:
        if not self.comp.all_evaluations:
            return None

        combined_text = "\n\n".join(
            f"Evaluator {name}:\n{text}"
            for name, text in self.comp.all_evaluations.items()
        )

        active_names = [
            self.comp.anon_mapping[k] for k in self.comp.active_model_keys
        ]

        return {
            name: {"Tournament Evaluation": combined_text}
            for name in active_names
        }

    async def _run_refinement_strategy(
        self,
        initial_question: str,
        round_num: int,
        evaluation_context: dict[str, dict[str, str]] | None = None,
    ) -> bool:
        return await self._run_improvement_phase(
            initial_question,
            config_key="refinement_phase",
            phase_name="Refinement Phase",
            answer_index=-1,
            round_num=round_num,
            evaluation_context=evaluation_context,
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
                "Preserved %s insights from %s",
                len(insights_preserved),
                eliminated_model,
            )

        # Get variance and consensus data for this elimination
        score_variance = None
        elimination_confidence = "unknown"
        if self.comp.model_score_variances:
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
            "reason": self.comp.elimination_reason
            or "Lowest score in evaluation",
            "score": self.comp.elimination_score,
            "score_variance": score_variance,
            "elimination_confidence": elimination_confidence,
            "insights_preserved": insights_preserved,  # Now populated!
        }

        # Add self-scoring bias if available
        if self.comp.self_scoring_biases:
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
            "ELIMINATED: %s - %s models remaining",
            eliminated_model,
            len(self.comp.active_model_keys),
        )
        self.logger.info("   Reason: %s", elimination_info["reason"])

    def _determine_elimination_confidence(self, score_variance: float) -> str:
        if score_variance < VARIANCE_HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        if score_variance < VARIANCE_MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"

    def _collect_all_final_responses(self) -> dict[str, str]:
        all_responses: dict[str, str] = {}
        for round_answers in self.comp.previous_answers:
            for model_name, response in round_answers.items():
                all_responses[model_name] = response
        return all_responses

    def _select_synthesis_model_key(self, champion_model_key: str) -> str:
        if (
            self.comp.judge_model_key
            and self.comp.judge_model_key in self.comp.models
        ):
            return self.comp.judge_model_key
        return champion_model_key

    async def _run_synthesis(
        self,
        initial_question: str,
        champion_model_key: str,
        champion_answer: str,
        km_context: str = "",
    ) -> str:
        if not self.comp.features.get("synthesis_enabled", True):
            return champion_answer

        all_responses = self._collect_all_final_responses()

        if len(all_responses) <= 1:
            return champion_answer

        self.logger.info(
            "SYNTHESIS PHASE: Combining insights from all participants",
            extra={"display_type": "section_header"},
        )

        kb_context = await self.comp.get_knowledge_bank_context()
        if km_context:
            kb_context = (
                f"{kb_context}\n\n{km_context}" if kb_context else km_context
            )

        synthesis_prompt = self.comp.prompt_builder.build_synthesis_prompt(
            initial_question, all_responses, kb_context
        )

        synthesis_model_key = self._select_synthesis_model_key(
            champion_model_key
        )
        response = await self.comp.execute_single_model_task(
            model_key=synthesis_model_key,
            prompt=synthesis_prompt,
            context_for_logging="SYNTHESIS",
        )

        if not response.is_error() and response.content.strip():
            self.logger.info(
                "Synthesis complete — combined %s perspectives",
                len(all_responses),
            )
            return response.content.strip()

        self.logger.warning("Synthesis failed, using champion answer")
        return champion_answer

    def _render_knowledge_map_context(self, km: Any) -> str:
        try:
            from certamen_core.domain.knowledge_map.renderer import (
                KnowledgeMapRenderer,
            )

            return KnowledgeMapRenderer().to_markdown(km)
        except Exception as e:
            self.logger.debug("KM context rendering failed: %s", e)
            return ""

    async def _save_knowledge_map_to_file(self, km: Any) -> None:
        try:
            from datetime import datetime

            from certamen_core.domain.knowledge_map.renderer import (
                KnowledgeMapRenderer,
            )

            km_md = KnowledgeMapRenderer().to_markdown(km)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            await self.comp.host.write_file(
                f"certamen_{timestamp}_knowledge_map.md", km_md
            )
            self.logger.info(
                "Knowledge map saved to certamen_%s_knowledge_map.md",
                timestamp,
            )
        except Exception as e:
            self.logger.warning("Failed to save knowledge map to file: %s", e)

    async def _finalize_with_champion(
        self,
        initial_question: str,
        final_model_key: str,
        final_model_anon: str,
        champion_answer: str,
    ) -> str:
        km = await self._build_knowledge_map(
            initial_question, champion_answer, final_model_key
        )

        km_context_str = (
            self._render_knowledge_map_context(km) if km is not None else ""
        )

        final_answer = await self._run_synthesis(
            initial_question,
            final_model_key,
            champion_answer,
            km_context=km_context_str,
        )

        if km is not None:
            km.synthesis = final_answer

        if self.comp.features.get("save_reports_to_disk", True):
            await self.comp.history_builder.save_champion_report(
                initial_question=initial_question,
                final_model_anon=final_model_anon,
                champion_answer=final_answer,
                all_previous_answers=self.comp.previous_answers,
            )

        self.logger.info(
            "Synthesized Final Answer",
            extra={"display_type": "section_header"},
        )
        self.logger.info(
            final_answer,
            extra={
                "display_type": "model_response",
                "model_name": "success",
            },
        )

        if km is not None:
            self._knowledge_map = km
            await self._persist_knowledge_map(km)
            await self._save_knowledge_map_to_file(km)

        return final_answer

    async def _finalize_tournament(self, initial_question: str) -> str:
        if len(self.comp.active_model_keys) == 0:
            self.logger.error(
                "All models failed during tournament. No champion can be determined."
            )
            return "Tournament ended prematurely: All models failed or were eliminated due to errors."

        final_model_key = self.comp.active_model_keys[0]
        final_model_anon = self.comp.anon_mapping[final_model_key]

        champion_answer = self.comp.previous_answers[-1].get(
            final_model_anon, ""
        )

        if not champion_answer:
            self.logger.error(
                "Could not find final answer for champion %s",
                final_model_anon,
            )
            return (
                f"Champion {final_model_anon} determined but answer not found."
            )

        self.logger.info(
            "CHAMPION: %s - Using their final refined answer",
            final_model_anon,
        )

        return await self._finalize_with_champion(
            initial_question,
            final_model_key,
            final_model_anon,
            champion_answer,
        )


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
        self.logger = get_contextual_logger("certamen.comparison")

        self.cost_tracker = CostTracker(logger=self.logger)

        self.retry_settings = config["retry"]
        self.features = config["features"]
        self.prompts = self._apply_confidence_calibration(
            config["prompts"], self.features
        )

        self.previous_answers: list[dict[str, str]] = []
        self.eliminated_models: list[dict[str, Any]] = []
        self.evaluation_history: list[dict[str, Any]] = []
        self.evaluation_scores: (
            dict[str, float] | dict[str, dict[str, float]]
        ) = {}
        self.all_evaluations: dict[str, str] = {}
        self.feedback_history: list[dict[str, Any]] = []
        self.criticism_history: list[dict[str, Any]] = []
        self.interrogation_context: str = ""

        self.elimination_reason: str = ""
        self.elimination_score: float | None = None
        self.model_score_variances: dict[str, dict[str, float | int]] = {}
        self.self_scoring_biases: dict[str, float] = {}
        self.current_leader_key: str | None = None

        deterministic_mode = self.features.get("deterministic_mode", False)
        if deterministic_mode:
            self.logger.info(
                "Running in deterministic mode with fixed random seed"
            )

        self.anonymizer = ModelAnonymizer(deterministic_mode)
        self.score_extractor = ScoreExtractor()
        self.report_generator = ReportGenerator(self.host)
        self.formatter = PromptFormatter()
        self.prompt_builder = PromptBuilder(
            self.prompts,
            self.formatter,
            reasoning_perspectives=config.get("reasoning_perspectives", []),
        )

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

        self.aggregator = ScoreAggregator(self)
        self.ranking_engine = RankingEngine(self)
        self.history_builder = TournamentHistoryBuilder(self)
        self.runner = TournamentRunner(self)

    @property
    def total_cost(self) -> float:
        return self.cost_tracker.total_cost

    @property
    def cost_by_model(self) -> dict[str, float]:
        return self.cost_tracker.cost_by_model

    @staticmethod
    def _apply_confidence_calibration(
        prompts: dict[str, Any], features: dict[str, Any]
    ) -> dict[str, Any]:
        if not features.get("confidence_calibration_enabled", True):
            return prompts
        calib_config = prompts.get("confidence_calibrated", {})
        calib_content = (
            calib_config.get("content", "")
            if isinstance(calib_config, dict)
            else ""
        )
        if not calib_content:
            return prompts
        initial_config = prompts.get("initial", {})
        if not isinstance(initial_config, dict):
            return prompts
        existing = initial_config.get("content", "")
        updated_initial = {
            **initial_config,
            "content": existing + "\n\n" + calib_content,
        }
        return {**prompts, "initial": updated_initial}

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
                "Judge model '%s' not found in available models",
                judge_model_config_key,
            )
            return None

        if judge_model_key in self.active_model_keys:
            self.logger.info(
                "Judge model '%s' will only act as judge and will not participate in the tournament.",
                self.models[judge_model_key].display_name,
                extra={"display_type": "colored_text", "color": "warning"},
            )
            self.active_model_keys.remove(judge_model_key)
            self.logger.info(
                "Removed judge model %s from tournament participants",
                judge_model_key,
            )

        return judge_model_key

    async def get_knowledge_bank_context(self) -> str:
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
                    "Filtered out placeholder/invalid response from %s: '%s'",
                    key,
                    txt,
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
            "Decoding shuffled names: %s mappings",
            len(reverse_shuffle_mapping),
        )
        decoded_text = text
        for code_name, orig_name in reverse_shuffle_mapping.items():
            pattern = r"\b" + re.escape(code_name) + r"\b"
            decoded_text = re.sub(pattern, f"({orig_name})", decoded_text)
            self.logger.debug("Decoded: %s → %s", code_name, orig_name)
        return decoded_text

    def _handle_model_failure(self, model_key: str, reason: str) -> None:
        display_name = self.anon_mapping.get(model_key, model_key)
        self.logger.error(
            "Removing %s from tournament: %s", display_name, reason
        )

        if model_key in self.active_model_keys:
            self.active_model_keys.remove(model_key)
        self.eliminated_models.append(
            {"model": display_name, "reason": f"Model failure: {reason}"}
        )
        self.anon_mapping.pop(model_key, None)

    async def execute_single_model_task(
        self,
        model_key: str,
        prompt: str,
        context_for_logging: str,
    ) -> ModelResponse:
        from certamen_core.shared.constants import DEFAULT_MODEL_TIMEOUT

        model = self.models[model_key]

        # Use task context for correlation IDs
        with self.logger.task_context(
            phase=context_for_logging, model=model_key
        ):
            self.logger.debug(
                "Preparing to execute task for model: %s", model_key
            )

            self.logger.debug(
                "Executing %s",
                context_for_logging,
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
                        "Starting async request with %ss timeout",
                        timeout_value,
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
                    "Task timeout for %s after %ss",
                    model.full_display_name,
                    DEFAULT_MODEL_TIMEOUT * 4,
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
                        "Added $%.4f for %s, total now: $%.4f",
                        response.cost,
                        model_display_name,
                        self.cost_tracker.total_cost,
                    )
                else:
                    self.logger.debug(
                        "Zero cost response from %s", model.display_name
                    )
            else:
                self.logger.debug(
                    "No cost attribute in response from %s", model.display_name
                )

            # Apply meta-commentary filtering for improvement and synthesis responses
            if (
                response
                and response.content
                and context_for_logging in ("IMPROVEMENT", "SYNTHESIS")
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
            task = self.execute_single_model_task(
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

        # Content is already cleaned by execute_single_model_task if needed
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
                "No valid responses for %s from any models.",
                context_for_logging,
            )
            self.logger.warning(
                "Tournament cannot continue. Active models remaining: %s",
                len(self.active_model_keys),
            )
            return False

        if len(valid_results) < 2 and context_for_logging in [
            "INITIAL",
            "IMPROVEMENT",
        ]:
            self.logger.warning(
                "Only %s model(s) responded. Tournament may end prematurely.",
                len(valid_results),
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
                "Gathering %s parallel %s tasks",
                len(tasks),
                context_for_logging,
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
                "Unexpected error during parallel %s calls: %s",
                context_for_logging,
                e,
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

        model_key_index = {k: i for i, k in enumerate(self.active_model_keys)}

        def build_initial_prompt(model_key: str, _model: BaseModel) -> str:
            idx = model_key_index[model_key]
            return self.prompt_builder.build_initial_prompt(
                initial_question, perspective_index=idx
            )

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
        round_number: int = 0,
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
                task = self.execute_single_model_task(
                    model_key=reviewer_key,
                    prompt=prompt,
                    context_for_logging="FEEDBACK",
                )
                tasks.append(task)
                task_metadata.append((reviewer_key, target_model))

        if tasks:
            self.logger.info(
                "Gathering %s parallel feedback tasks", len(tasks)
            )
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                reviewer_key, target_model = task_metadata[i]
                reviewer_name = self.anon_mapping[reviewer_key]

                if isinstance(response, Exception):
                    self.logger.error(
                        "Error getting feedback from %s for %s: %s",
                        reviewer_name,
                        target_model,
                        response,
                    )
                    continue

                if not isinstance(response, ModelResponse):
                    continue

                if response.is_error():
                    self.logger.error(
                        "Error getting feedback from %s for %s: %s",
                        reviewer_name,
                        target_model,
                        response.error,
                    )
                    continue

                feedback_text = response.content
                feedback_context[target_model][reviewer_name] = feedback_text

                log_message = self.formatter.format_feedback_log(
                    reviewer_name, target_model, feedback_text
                )
                self.logger.debug(indent_text(log_message))
                self.logger.info(
                    "\n%s's feedback for %s:",
                    reviewer_name,
                    target_model,
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
            {"round": round_number, "feedback": feedback_context}
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
        kb_context = await self.get_knowledge_bank_context()

        if self.interrogation_context:
            interrogation_section = (
                "\n=== INTERROGATION FINDINGS: Knowledge surfaced through cross-examination ===\n"
                + self.interrogation_context[:3000]
                + "\n=== END INTERROGATION FINDINGS ===\n"
            )
            kb_context = (
                kb_context + interrogation_section
                if kb_context
                else interrogation_section
            )

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
            # nosemgrep: python-logger-credential-disclosure
            self.logger.info(
                "Selected %s as emergency judge (context_window=%s, max_tokens=%s)",
                model.display_name,
                model.context_window,
                model.max_tokens,
            )

        return best_model_key

    def _peer_scores_are_valid(self) -> bool:
        if not self.evaluation_scores:
            return False
        first_value = next(iter(self.evaluation_scores.values()), None)
        if not isinstance(first_value, dict):
            return False
        return any(self.evaluation_scores.values())

    async def _peer_evaluation_with_emergency_fallback(
        self,
        initial_question: str,
        responses: dict[str, str],
    ) -> dict[str, str]:
        result = await self._run_peer_evaluation(initial_question, responses)

        if self._peer_scores_are_valid():
            return result

        self.logger.warning(
            "ALL peer evaluators failed to provide valid scores. "
            "Falling back to JUDGE MODE with largest model."
        )
        emergency_judge = self._select_largest_model_as_judge()
        if emergency_judge:
            self.logger.info(
                "EMERGENCY JUDGE MODE: Using %s",
                self.models[emergency_judge].display_name,
            )
            return await self._run_judge_evaluation(
                emergency_judge, initial_question, responses
            )

        self.logger.error(
            "CRITICAL: No model available for emergency judge fallback"
        )
        return result

    async def run_cross_evaluation(
        self,
        initial_question: str,
        responses: dict[str, str],
        round_num: int,
    ) -> dict[str, str]:
        self.logger.info(
            "Phase 3: Cross-Evaluation (Round %s)",
            round_num,
            extra={"display_type": "section_header"},
        )

        if self.judge_model_key:
            self.logger.info(
                "Using dedicated judge model for evaluation: %s",
                self.judge_model_key,
            )
            return await self._run_judge_evaluation_with_fallback(
                self.judge_model_key, initial_question, responses
            )

        self.logger.info(
            "Using cross-evaluation (peer review). All models will evaluate each other."
        )
        return await self._peer_evaluation_with_emergency_fallback(
            initial_question, responses
        )

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
            "Primary judge %s failed. Attempting fallback strategies...",
            self.models[primary_judge_key].display_name,
        )

        # Fallback 1: Try emergency judge (largest model)
        emergency_judge = self._select_largest_model_as_judge()
        if emergency_judge and emergency_judge != primary_judge_key:
            self.logger.info(
                "Fallback 1: Trying emergency judge %s",
                self.models[emergency_judge].display_name,
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
            "Using %s as the judge.",
            judge_display_name,
            extra={"display_type": "colored_text", "color": "info"},
        )

        shuffled_responses = collaborative_responses
        reverse_shuffle_mapping = {
            v: self.models[k].full_display_name
            for k, v in self.anon_mapping.items()
            if k in self.models
        }

        self.logger.info("Judge %s will evaluate:", judge_display_name)
        for anon_name, real_name in reverse_shuffle_mapping.items():
            self.logger.info("  %s (actually %s)", anon_name, real_name)

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

        response = await self.execute_single_model_task(
            model_key=judge_model_key,
            prompt=prompt,
            context_for_logging="JUDGE_EVAL",
        )

        if response.is_error():
            self.logger.warning(
                "Judge %s evaluation failed: %s",
                judge_display_name,
                response.error,
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
            "\nEvaluation from Judge %s:",
            judge_display_name,
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
            "Judge %s provided scores: %s", judge_display_name, raw_scores
        )
        return self.all_evaluations

    async def _run_peer_evaluation(
        self,
        initial_question: str,
        collaborative_responses: dict[str, str],
    ) -> dict[str, str]:
        all_evaluations = {}
        evaluation_scores = {}

        shuffled_responses = collaborative_responses
        reverse_shuffle_mapping = {
            v: self.models[k].full_display_name
            for k, v in self.anon_mapping.items()
            if k in self.models
        }

        self.logger.info("Peer evaluation anonymization mapping:")
        for anon_name, real_name in reverse_shuffle_mapping.items():
            self.logger.info("  %s = %s", anon_name, real_name)

        def get_responses_for_evaluator(
            evaluator_anon_name: str,
        ) -> dict[str, str]:
            self.logger.debug(
                "Evaluator %s will evaluate %s responses: %s",
                evaluator_anon_name,
                len(shuffled_responses),
                list(shuffled_responses.keys()),
            )
            return shuffled_responses

        def build_peer_review_prompt(model_key: str, model: BaseModel) -> str:
            evaluator_anon_name = self.anon_mapping[model_key]
            evaluator_display_name = self.models[model_key].display_name

            self.logger.info(
                "Evaluator %s (as %s) is scoring:",
                evaluator_display_name,
                evaluator_anon_name,
            )
            for anon_name, real_name in reverse_shuffle_mapping.items():
                self.logger.info(
                    "  %s (labeled as %s in prompt, actually %s)",
                    anon_name,
                    anon_name,
                    real_name,
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
                "Building evaluation prompt for %s. "
                "Formatted responses length: %s, "
                "Models to evaluate: %s",
                evaluator_anon_name,
                len(formatted_responses),
                code_names_to_evaluate,
            )
            prompt = self.prompt_builder.build_evaluation_prompt(
                initial_question,
                formatted_responses,
                code_names_to_evaluate,
            )
            self.logger.debug(
                "Full evaluation prompt for %s:\n%s",
                evaluator_anon_name,
                prompt,
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

            # Keep scores with anonymized keys - get_aggregated_scores() expects them
            # raw_scores already has anonymized keys (LLM1, LLM2, etc.) from code_names
            if raw_scores:
                self.logger.info(
                    "Extracted scores from %s: %s", display_name, raw_scores
                )

            evaluation_scores[display_name] = raw_scores
            decoded_eval = self._decode_shuffled_names(
                evaluation_text, reverse_shuffle_mapping
            )

            all_evaluations[display_name] = decoded_eval
            self.logger.info(
                "\nEvaluations from %s:",
                display_name,
                extra={"display_type": "colored_text", "color": display_name},
            )
            self.logger.info(
                decoded_eval,
                extra={"display_type": "colored_text", "color": display_name},
            )

        self.evaluation_scores = evaluation_scores
        self.all_evaluations = all_evaluations
        return all_evaluations

    def determine_lowest_and_highest_ranked_models(
        self,
    ) -> tuple[str, str]:
        return self.ranking_engine.determine_lowest_and_highest_ranked_models()

    def _display_cost_summary(self) -> None:
        self.cost_tracker.display_summary()

    async def run(self, initial_question: str) -> str:
        try:
            return await self.runner.run(initial_question)
        finally:
            self._display_cost_summary()
            self.logger.info(
                "Certamen Framework Tournament Complete",
                extra={"display_type": "section_header"},
            )
            # self.display.reset() # This needs to be handled by the event handler

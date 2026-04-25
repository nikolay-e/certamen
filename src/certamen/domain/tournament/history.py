from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from certamen.domain.tournament.tournament import ModelComparison


class TournamentHistoryBuilder:
    def __init__(self, comparison_instance: "ModelComparison") -> None:
        self.comp = comparison_instance
        self.logger = comparison_instance.logger

    def _get_phase_2_feedback(self) -> Any | None:
        if not self.comp.feedback_history:
            return None
        for feedback_entry in self.comp.feedback_history:
            if feedback_entry.get("round") == 0:
                return feedback_entry.get("feedback")
        return None

    def _build_phase_2_data(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        phase_2_feedback = self._get_phase_2_feedback()

        phase_2_data: dict[str, Any] = {}

        improved_answers = all_previous_answers[1].copy()

        if phase_2_feedback:
            phase_2_data["feedback"] = phase_2_feedback
            phase_2_data["enhanced_answers"] = improved_answers
            return {
                "Phase 2: Positive Reinforcement & Strength Amplification": phase_2_data
            }
        else:
            return {"Phase 2: Collaborative Analysis": improved_answers}

    def _add_round_evaluations(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        if elimination_round - 1 < len(self.comp.evaluation_history):
            eval_data = self.comp.evaluation_history[elimination_round - 1]
            round_data["evaluations"] = eval_data.get("evaluations", {})
            round_data["scores"] = eval_data.get("scores", {})

    def _add_round_feedback(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        for feedback_entry in self.comp.feedback_history:
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
            self._add_round_feedback(round_data, elimination_round)
            round_data["refined_answers"] = all_previous_answers[i].copy()
            tournament_data[f"Elimination Round {elimination_round}"] = (
                round_data
            )
        return tournament_data

    def build_tournament_history(
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

    async def save_champion_report(
        self,
        initial_question: str,
        final_model_anon: str,
        champion_answer: str,
        all_previous_answers: list[dict[str, str]],
    ) -> None:
        from certamen.domain.reporting.provenance import ProvenanceReport

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tournament_history = self.build_tournament_history(
            all_previous_answers
        )

        if self.comp.active_model_keys:
            champion_display = self.comp.models[
                self.comp.active_model_keys[0]
            ].full_display_name
        else:
            champion_display = "unknown"
        champion_model = f"{final_model_anon} (model: {champion_display})"

        tournament_data = {
            **self.comp.cost_tracker.get_summary(),
            "eliminated_models": self.comp.eliminated_models.copy(),
            "complete_tournament_history": tournament_history,
        }

        provenance = ProvenanceReport(
            question=initial_question,
            champion_model=champion_model,
            champion_answer=champion_answer,
            tournament_data=tournament_data,
        )

        outputs_dir = str(self.comp.host.base_dir)
        saved_paths = await provenance.save_to_file(
            outputs_dir, timestamp, write_file=self.comp.host.write_file
        )

        for file_type, file_path in saved_paths.items():
            self.logger.info("%s: %s", file_type, file_path)

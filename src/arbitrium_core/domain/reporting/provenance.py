import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from arbitrium_core.shared.json_utils import sanitize_for_json


class ProvenanceReport:
    def __init__(
        self,
        question: str,
        champion_model: str,
        champion_answer: str,
        tournament_data: dict[str, Any],
    ):
        self.question = question
        self.champion_model = champion_model
        self.champion_answer = champion_answer
        self.tournament_data = tournament_data
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _generate_champion_with_provenance(self) -> str:
        md = []

        # Header
        md.append(f"# Champion Solution {self.timestamp}")
        md.append("")
        md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")

        # Question (fenced to prevent markdown structure breaking)
        md.append("## Initial Question")
        md.append("")
        md.append("```")
        md.append(self.question)
        md.append("```")
        md.append("")

        # Champion info
        md.append("## Champion Model")
        md.append("")
        md.append(self.champion_model)
        md.append("")

        # Champion solution (FULL, no truncation, fenced to prevent markdown structure breaking)
        md.append("## Champion Solution")
        md.append("")
        md.append("```markdown")
        md.append(self.champion_answer)
        md.append("```")
        md.append("")

        # Separator
        md.append("---")
        md.append("")

        # Tournament Provenance (evolution, NOT full history)
        md.append("## Tournament Provenance")
        md.append("")
        md.append(self.generate_markdown())

        return "\n".join(md)

    def _generate_provenance_metadata(self) -> dict[str, Any]:
        report = {
            "tournament_id": self.timestamp,
            "question": self._sanitize_for_json(self.question),
            "champion_model": self._sanitize_for_json(self.champion_model),
            "final_answer": self._sanitize_for_json(self.champion_answer),
            "phases": self._extract_phases(),  # Already sanitized in _extract_phases
            "eliminations": self._sanitize_for_json(
                self._extract_eliminations()
            ),
            "cost_summary": self._sanitize_for_json(
                self.tournament_data.get("cost_summary", {})
            ),
            "generated_at": datetime.now().isoformat(),
        }
        return report

    def _format_phase_initial(
        self, md: list[str], phase: dict[str, Any]
    ) -> None:
        md.append("All models provided initial responses independently.")
        md.append("")
        for model, answer in phase.get("responses", {}).items():
            md.append(f"**{model}:**")
            md.append("")
            # Use fenced code block to protect MD content inside answers
            md.append("```")
            md.append(str(answer))
            md.append("```")
            md.append("")

    def _format_phase_improvement(
        self, md: list[str], phase: dict[str, Any]
    ) -> None:
        md.append(
            f"Models improved their answers based on {phase.get('strategy', 'feedback')}."
        )
        md.append("")

        # Show criticisms if available
        if "criticisms" in phase and phase["criticisms"] is not None:
            md.append("#### Critiques Exchanged")
            md.append("")
            for target, critics in phase["criticisms"].items():
                md.append(f"{target} received feedback from:")
                for critic_model, critique in critics.items():
                    md.append(f"- *{critic_model}:* {critique}")
                md.append("")

        # Show positive feedback if available
        if "feedback" in phase and phase["feedback"] is not None:
            md.append("#### Positive Feedback Exchanged")
            md.append("")
            for target, supporters in phase["feedback"].items():
                md.append(f"{target} received support from:")
                for supporter, feedback in supporters.items():
                    md.append(f"- *{supporter}:* {feedback}")
                md.append("")

    def _format_phase_evaluation(
        self, md: list[str], phase: dict[str, Any]
    ) -> None:
        md.append("Models were evaluated and ranked.")
        md.append("")

        scores = phase.get("scores", {})
        if scores:
            md.append("Scores:")
            md.append("")
            # Filter out non-numeric scores and sort
            numeric_scores = {
                model: score
                for model, score in scores.items()
                if isinstance(score, (int, float))
            }
            if numeric_scores:
                for model, score in sorted(
                    numeric_scores.items(), key=lambda x: x[1], reverse=True
                ):
                    md.append(f"- {model}: {score:.2f}/10")
                md.append("")

    def _format_eliminations(self, md: list[str]) -> None:
        eliminations = self._extract_eliminations()
        if not eliminations:
            return

        md.append("### Eliminations")
        md.append("")

        for elim in eliminations:
            md.append(
                f"#### Round {elim['round']}: {elim['model']} Eliminated"
            )
            md.append("")
            md.append(f"**Score:** {elim.get('score', 'N/A')}")
            md.append("")
            if "reason" in elim:
                md.append(f"**Reason:** {elim['reason']}")
                md.append("")

            # ALL insights, no limit
            if elim.get("insights_preserved"):
                md.append("**Insights Preserved from this Model:**")
                md.append("")
                for insight in elim["insights_preserved"]:
                    md.append(f"- {insight}")
                md.append("")

            md.append("---")
            md.append("")

    def _format_cost_summary(self, md: list[str]) -> None:
        cost = self.tournament_data.get("total_cost")
        if not cost:
            return

        md.append("### Cost Summary")
        md.append("")
        # Handle cost as either float or string
        if isinstance(cost, str):
            md.append(f"**Total:** {cost}")
        else:
            md.append(f"**Total:** ${cost:.4f}")
        md.append("")

        cost_by_model = self.tournament_data.get("cost_by_model", {})
        if cost_by_model:
            md.append("**Cost by Model:**")
            md.append("")

            def get_numeric_cost(item: tuple[str, Any]) -> float:
                cost_str = item[1]
                if isinstance(cost_str, str):
                    return float(cost_str.replace("$", ""))
                return float(cost_str)

            for model, model_cost in sorted(
                cost_by_model.items(), key=get_numeric_cost, reverse=True
            ):
                md.append(f"- {model}: {model_cost}")
            md.append("")

    def generate_markdown(self) -> str:
        md = []

        # Tournament Evolution
        md.append("### Tournament Evolution")
        md.append("")

        phases = self._extract_phases()
        for phase in phases:
            md.append(f"#### {phase['name']}")
            md.append("")

            if phase["type"] == "initial":
                self._format_phase_initial(md, phase)
            elif phase["type"] == "improvement":
                self._format_phase_improvement(md, phase)
            elif phase["type"] == "evaluation":
                self._format_phase_evaluation(md, phase)

            md.append("---")
            md.append("")

        # Eliminations
        self._format_eliminations(md)

        # Cost Summary
        self._format_cost_summary(md)

        return "\n".join(md)

    def _determine_strategy(self, phase_name: str) -> str:
        if "Criticism" in phase_name:
            return "cross-criticism"
        elif "Positive" in phase_name:
            return "positive reinforcement"
        else:
            return "collaborative"

    def _sanitize_for_json(self, obj: Any) -> Any:
        return sanitize_for_json(obj)

    def _extract_initial_phase(
        self, phase_name: str, phase_data: Any
    ) -> dict[str, Any]:
        return {
            "name": phase_name,
            "type": "initial",
            "responses": (
                self._sanitize_for_json(phase_data)
                if isinstance(phase_data, dict)
                else {}
            ),
        }

    def _extract_improvement_phase(
        self, phase_name: str, phase_data: Any
    ) -> dict[str, Any]:
        if not isinstance(phase_data, dict):
            return {
                "name": phase_name,
                "type": "improvement",
                "strategy": self._determine_strategy(phase_name),
            }

        # Check if it's structured or old format
        if not self._has_structured_improvement_data(phase_data):
            return {
                "name": phase_name,
                "type": "improvement",
                "strategy": self._determine_strategy(phase_name),
                "improved_answers": self._sanitize_for_json(phase_data),
            }

        # Build structured phase info
        phase_info: dict[str, Any] = {
            "name": phase_name,
            "type": "improvement",
            "strategy": self._determine_strategy(phase_name),
        }

        # Add optional fields
        self._add_optional_field(phase_info, phase_data, "criticisms")
        self._add_optional_field(phase_info, phase_data, "feedback")

        # Add improved answers (key data)
        improved_answers = phase_data.get(
            "improved_answers"
        ) or phase_data.get("enhanced_answers")
        if improved_answers:
            phase_info["improved_answers"] = self._sanitize_for_json(
                improved_answers
            )

        return phase_info

    def _extract_evaluation_phase(
        self, phase_name: str, phase_data: Any
    ) -> dict[str, Any]:
        if not isinstance(phase_data, dict):
            return {
                "name": phase_name,
                "type": "evaluation",
                "scores": {},
                "evaluations": {},
                "refined_answers": {},
            }

        return {
            "name": phase_name,
            "type": "evaluation",
            "scores": self._sanitize_for_json(phase_data.get("scores", {})),
            "evaluations": self._sanitize_for_json(
                phase_data.get("evaluations", {})
            ),
            "refined_answers": self._sanitize_for_json(
                phase_data.get("refined_answers", {})
            ),
        }

    def _has_structured_improvement_data(
        self, phase_data: dict[str, Any]
    ) -> bool:
        return any(
            k in phase_data
            for k in [
                "criticisms",
                "feedback",
                "improved_answers",
                "enhanced_answers",
            ]
        )

    def _add_optional_field(
        self,
        phase_info: dict[str, Any],
        phase_data: dict[str, Any],
        field_name: str,
    ) -> None:
        if field_name in phase_data and phase_data[field_name] is not None:
            phase_info[field_name] = self._sanitize_for_json(
                phase_data[field_name]
            )

    def _is_initial_phase(self, phase_name: str) -> bool:
        return "Initial" in phase_name

    def _is_improvement_phase(self, phase_name: str) -> bool:
        return any(
            keyword in phase_name
            for keyword in [
                "Cross-Criticism",
                "Positive Reinforcement",
                "Collaborative",
            ]
        )

    def _is_evaluation_phase(self, phase_name: str) -> bool:
        return "Elimination Round" in phase_name

    def _extract_phases(self) -> list[dict[str, Any]]:
        phases = []
        history = self.tournament_data.get("complete_tournament_history", {})

        for phase_name, phase_data in history.items():
            if self._is_initial_phase(phase_name):
                phases.append(
                    self._extract_initial_phase(phase_name, phase_data)
                )
            elif self._is_improvement_phase(phase_name):
                phases.append(
                    self._extract_improvement_phase(phase_name, phase_data)
                )
            elif self._is_evaluation_phase(phase_name):
                phases.append(
                    self._extract_evaluation_phase(phase_name, phase_data)
                )

        return phases

    def _extract_eliminations(self) -> list[dict[str, Any]]:
        eliminations = []
        eliminated_models = self.tournament_data.get("eliminated_models", [])

        for i, elim_data in enumerate(eliminated_models, 1):
            if isinstance(elim_data, dict):
                eliminations.append(
                    {
                        "round": i,
                        "model": elim_data.get("model", "Unknown"),
                        "score": elim_data.get("score"),
                        "reason": elim_data.get(
                            "reason", "Lowest score in evaluation"
                        ),
                        "insights_preserved": elim_data.get(
                            "insights_preserved", []
                        ),
                    }
                )
            else:
                # Legacy format: just model name
                eliminations.append(
                    {
                        "round": i,
                        "model": str(elim_data),
                        "score": None,
                        "reason": "Lowest score in evaluation",
                        "insights_preserved": [],
                    }
                )

        return eliminations

    async def save_to_file(
        self, output_dir: str, timestamp: str
    ) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Save champion solution MD with provenance at the end
        champion_md_filename = f"arbitrium_{timestamp}_champion_solution.md"
        champion_md_path = output_path / champion_md_filename

        champion_md_content = self._generate_champion_with_provenance()
        await asyncio.to_thread(
            lambda: champion_md_path.write_text(
                champion_md_content, encoding="utf-8"
            )
        )

        # 2. Save compact provenance JSON (metadata only, no full history)
        provenance_json_filename = f"arbitrium_{timestamp}_provenance.json"
        provenance_json_path = output_path / provenance_json_filename

        provenance_data = self._generate_provenance_metadata()
        await asyncio.to_thread(
            lambda: provenance_json_path.write_text(
                json.dumps(provenance_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        )

        # 3. Save complete history JSON (all tournament data)
        complete_history_filename = (
            f"arbitrium_{timestamp}_complete_history.json"
        )
        complete_history_path = output_path / complete_history_filename

        await asyncio.to_thread(
            lambda: complete_history_path.write_text(
                json.dumps(self.tournament_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        )

        return {
            "champion_md": str(champion_md_path),
            "provenance_json": str(provenance_json_path),
            "complete_history_json": str(complete_history_path),
        }

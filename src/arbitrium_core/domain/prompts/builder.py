"""Prompt building for Arbitrium Framework."""

from typing import Any

from arbitrium_core.domain.prompts.formatter import PromptFormatter
from arbitrium_core.domain.prompts.templates import (
    EVALUATION_PROMPT_TEMPLATE,
    FEEDBACK_PROMPT_TEMPLATE,
    IMPROVEMENT_PROMPT_TEMPLATE,
    INITIAL_PROMPT_TEMPLATE,
)
from arbitrium_core.ports.llm import BaseModel


class PromptBuilder:
    """Builds prompts for the different phases of the tournament."""

    def __init__(
        self,
        prompts: dict[str, dict[str, Any]],
        formatter: PromptFormatter | None = None,
    ):
        self.prompts = prompts
        self.formatter = formatter or PromptFormatter()

    def _format_prompt(self, prompt_type: str, context: dict[str, Any]) -> str:
        prompt_config = self.prompts.get(prompt_type)
        if not prompt_config:
            raise ValueError(
                f"Prompt type '{prompt_type}' not found in config. Available: {list(self.prompts.keys())}"
            )

        # Extract content from structured format
        prompt_content = prompt_config.get("content", "")
        if isinstance(prompt_content, str):
            return prompt_content.format(**context)
        return str(prompt_content)

    def _build_feedback_context(
        self,
        improvement_context: dict[str, dict[str, str]] | None,
        display_name: str,
    ) -> str:
        if not improvement_context:
            return ""
        feedbacks = improvement_context.get(display_name, {})
        if not feedbacks:
            return ""
        return "\n\n".join(
            self.formatter.format_feedback_wrapper(reviewer, text)
            for reviewer, text in feedbacks.items()
        )

    def _build_other_responses_context(
        self,
        other_responses: dict[str, str] | None,
        display_name: str,
    ) -> str:
        if not other_responses:
            return ""
        filtered = {
            k: v for k, v in other_responses.items() if k != display_name
        }
        if not filtered:
            return ""

        responses = []
        for name, resp in filtered.items():
            responses.append(
                self.formatter.format_response_wrapper(name, resp)
            )

        return "\n\n".join(responses)

    def _build_full_improvement_context(
        self, context_text: str, other_responses_text: str
    ) -> str:
        """Combine feedback and other responses into full context.

        Args:
            context_text: Formatted feedback text
            other_responses_text: Formatted other responses text

        Returns:
            Combined context with section wrappers
        """
        parts: list[str] = []
        if context_text:
            parts.append(self.formatter.wrap_section("FEEDBACK", context_text))
        if other_responses_text:
            parts.append(
                self.formatter.wrap_section(
                    "OTHER RESPONSES", other_responses_text
                )
            )
        return "\n\n".join(parts)

    def build_initial_prompt(self, initial_question: str) -> str:
        """Build the initial prompt for the first round.

        Args:
            initial_question: The question to answer

        Returns:
            Formatted initial prompt
        """
        base_prompt = self._format_prompt("initial", context={})
        question_section = self.formatter.wrap_section(
            "QUESTION", initial_question
        )

        return INITIAL_PROMPT_TEMPLATE.format(
            base_prompt=base_prompt, question_section=question_section
        )

    def build_feedback_prompt(
        self,
        initial_question: str,
        target_answer: str,
        feedback_instruction: str,
    ) -> str:
        """Build the prompt for the feedback phase.

        Args:
            initial_question: The original question
            target_answer: Answer to provide feedback on
            feedback_instruction: Instructions for giving feedback

        Returns:
            Formatted feedback prompt
        """
        base_prompt = self._format_prompt("feedback", context={})
        question_section = self.formatter.wrap_section(
            "QUESTION", initial_question
        )
        answer_section = self.formatter.wrap_section("ANSWER", target_answer)

        return FEEDBACK_PROMPT_TEMPLATE.format(
            feedback_instruction=feedback_instruction,
            base_prompt=base_prompt,
            question_section=question_section,
            answer_section=answer_section,
        )

    def build_improvement_prompt(
        self,
        initial_question: str,
        own_answer: str,
        improvement_instruction: str,
        kb_context: str,
        improvement_context: dict[str, dict[str, str]] | None,
        other_responses: dict[str, str] | None,
        model: BaseModel,
        display_name: str,
    ) -> str:
        """Build the prompt for the improvement phase.

        Args:
            initial_question: The original question
            own_answer: Model's own answer to improve
            improvement_instruction: Instructions for improvement
            kb_context: Knowledge bank context text
            improvement_context: Feedback from other models
            other_responses: Other models' responses
            model: The model instance (for future customization)
            display_name: Display name of the model

        Returns:
            Formatted improvement prompt
        """
        context_text = self._build_feedback_context(
            improvement_context, display_name
        )
        other_responses_text = self._build_other_responses_context(
            other_responses, display_name
        )
        full_context = self._build_full_improvement_context(
            context_text, other_responses_text
        )

        base_prompt = self._format_prompt("improvement", context={})
        question_section = self.formatter.wrap_section(
            "QUESTION", initial_question
        )
        answer_section = self.formatter.wrap_section("ANSWER", own_answer)

        context_section = f"\n\n{full_context}" if full_context else ""
        knowledge_section = (
            f"\n\n{self.formatter.wrap_section('KNOWLEDGE BANK', kb_context)}"
            if kb_context
            else ""
        )

        return IMPROVEMENT_PROMPT_TEMPLATE.format(
            improvement_instruction=improvement_instruction,
            base_prompt=base_prompt,
            question_section=question_section,
            answer_section=answer_section,
            context_section=context_section,
            knowledge_section=knowledge_section,
        )

    def build_evaluation_prompt(
        self,
        initial_question: str,
        formatted_responses: str,
        model_names: list[str] | None = None,
    ) -> str:
        """Builds the detailed prompt for the evaluation phase.

        Args:
            initial_question: The original question
            formatted_responses: Pre-formatted responses string
            model_names: List of model names to evaluate

        Returns:
            Formatted evaluation prompt
        """
        base_prompt = self._format_prompt("evaluate", context={})
        question_section = self.formatter.wrap_section(
            "QUESTION", initial_question
        )
        responses_section = self.formatter.wrap_section(
            "RESPONSES", formatted_responses
        )

        # Generate a list of models that need to be scored (for clarity)
        models_list = (
            "\n".join([f"- {name}" for name in sorted(model_names)])
            if model_names
            else "- LLM1\n- LLM2\n- LLM3"
        )

        return EVALUATION_PROMPT_TEMPLATE.format(
            base_prompt=base_prompt,
            question_section=question_section,
            responses_section=responses_section,
            model_names=models_list,
        )

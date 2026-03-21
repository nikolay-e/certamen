"""
Markdown sanitization and processing utilities for Certamen Framework.
"""

import re
from typing import Any


def sanitize_for_markdown(text: str, preserve_markdown: bool = True) -> str:
    if not text:
        return ""

    # Only handle critical characters
    # 1. HTML entities for safety
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2. Escape headers only if we don't want to preserve markdown
    if not preserve_markdown:
        text = re.sub(r"^(#{1,6})\s", r"\\\1 ", text, flags=re.MULTILINE)

    return text


def sanitize_content_dict(
    content: dict[str, Any], preserve_markdown: bool = True
) -> dict[str, str]:
    sanitized_content = {}

    for key, value in content.items():
        if isinstance(value, str):
            sanitized_content[key] = sanitize_for_markdown(
                value, preserve_markdown
            )
        elif isinstance(value, dict):
            # Special handling for tournament_history - format as readable markdown
            if key == "tournament_history":
                sanitized_content[key] = format_tournament_history(value)
            else:
                sanitized_content[key] = sanitize_for_markdown(
                    str(value), preserve_markdown
                )
        else:
            # Convert non-string values to string and sanitize
            sanitized_content[key] = sanitize_for_markdown(
                str(value), preserve_markdown
            )

    return sanitized_content


def _format_nested_feedback(
    phase_data: dict[str, Any],
    formatted_sections: list[str],
    title: str,
    key: str,
) -> None:
    if phase_data.get(key):
        formatted_sections.append(f"#### {title}")
        for target_model, sources in phase_data[key].items():
            formatted_sections.append(f"**{title} for {target_model}:**")
            for source_name, text in sources.items():
                formatted_sections.append(f"*From {source_name}:*")
                formatted_sections.append(f"{text}")
                formatted_sections.append("")
            formatted_sections.append("")


def _format_evaluations(
    phase_data: dict[str, Any], formatted_sections: list[str]
) -> None:
    """Format cross-evaluations."""
    if phase_data.get("evaluations"):
        formatted_sections.append("#### Cross-Evaluations")
        for evaluator, evaluation_text in phase_data["evaluations"].items():
            formatted_sections.append(f"**{evaluator}'s Evaluation:**")
            formatted_sections.append(f"{evaluation_text}")
            formatted_sections.append("")


def _format_scores(
    phase_data: dict[str, Any], formatted_sections: list[str]
) -> None:
    """Format scores (peer review or single judge)."""
    if "scores" not in phase_data or not phase_data["scores"]:
        return

    formatted_sections.append("#### Scores")
    scores = phase_data["scores"]

    # Check if peer review (nested dict) or single judge (flat dict)
    if scores and isinstance(next(iter(scores.values())), dict):
        # Peer review format
        for evaluator, model_scores in scores.items():
            formatted_sections.append(f"**{evaluator}:**")
            for model, score in model_scores.items():
                formatted_sections.append(f"- {model}: {score:.2f}/10")
            formatted_sections.append("")
    else:
        # Single judge format
        for model, score in scores.items():
            formatted_sections.append(f"- {model}: {score:.2f}/10")
        formatted_sections.append("")


def _format_model_responses(
    phase_data: dict[str, Any],
    formatted_sections: list[str],
    title: str,
    key: str,
) -> None:
    """Format model responses (improved/enhanced/refined answers)."""
    if phase_data.get(key):
        formatted_sections.append(f"#### {title}")
        for model_name, response in phase_data[key].items():
            formatted_sections.append(f"**{model_name}:**")
            formatted_sections.append(f"{response}")
            formatted_sections.append("")


def _is_elimination_round(phase_data: dict[str, Any]) -> bool:
    """Check if phase_data is elimination round format."""
    special_keys = [
        "evaluations",
        "scores",
        "refined_answers",
        "criticisms",
        "improved_answers",
        "feedback",
        "enhanced_answers",
    ]
    return any(key in phase_data for key in special_keys)


def format_tournament_history(history: dict[str, dict[str, str]]) -> str:
    """
    Format tournament history as readable markdown.

    Args:
        history: Dictionary mapping phase names to either:
            - Dict of model responses (for initial/collaborative phases)
            - Dict with 'evaluations', 'scores', 'refined_answers' (for elimination rounds)

    Returns:
        Markdown-formatted tournament history
    """
    if not history:
        return "No tournament history available."

    formatted_sections = []

    for phase_name, phase_data in history.items():
        # Add phase header
        formatted_sections.append(f"### {phase_name}")

        if not phase_data:
            formatted_sections.append("*No data recorded for this phase.*")
            continue

        # Check if this is elimination round format (has nested structure)
        if _is_elimination_round(phase_data):
            # Elimination round or cross-criticism or positive reinforcement format
            _format_nested_feedback(
                phase_data, formatted_sections, "Positive Feedback", "feedback"
            )
            _format_nested_feedback(
                phase_data,
                formatted_sections,
                "Cross-Criticisms",
                "criticisms",
            )
            _format_evaluations(phase_data, formatted_sections)
            _format_scores(phase_data, formatted_sections)
            _format_model_responses(
                phase_data,
                formatted_sections,
                "Enhanced Answers After Strength Amplification",
                "enhanced_answers",
            )
            _format_model_responses(
                phase_data,
                formatted_sections,
                "Improved Answers After Self-Correction",
                "improved_answers",
            )

            # Show refined answers if present (from progressive refinement)
            _format_model_responses(
                phase_data,
                formatted_sections,
                "Refined Answers",
                "refined_answers",
            )
        else:
            # Simple format: just model responses
            for model_name, response in phase_data.items():
                formatted_sections.append(f"#### {model_name}")
                formatted_sections.append(f"{response}")
                formatted_sections.append("")  # Add spacing between models

    return "\n\n".join(formatted_sections)


def adjust_markdown_headers(content: str, start_level: int = 1) -> str:
    """
    Adjust all markdown headers in content to start at a specific level.

    Args:
        content: The markdown content to adjust
        start_level: The level to start headers at (1 = H1, 2 = H2, etc.)

    Returns:
        Content with adjusted header levels
    """
    if not content:
        return content

    lines = content.split("\n")
    adjusted_lines = []

    for line in lines:
        # Match markdown headers (# ## ### etc.)
        header_match = re.match(r"^(#+)\s+(.+)$", line)
        if header_match:
            current_hashes = header_match.group(1)
            header_text = header_match.group(2)
            current_level = len(current_hashes)

            # Adjust to start at the specified level
            new_level = current_level + start_level - 1
            new_hashes = "#" * new_level
            adjusted_lines.append(f"{new_hashes} {header_text}")
        else:
            adjusted_lines.append(line)

    return "\n".join(adjusted_lines)

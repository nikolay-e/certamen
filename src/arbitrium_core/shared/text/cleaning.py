import re
from typing import Any


def indent_text(text: str, indent: str = "    ") -> str:
    lines = [indent + line for line in text.splitlines() if line.strip()]
    return "\n" + "\n".join(lines)


def _compile_meta_patterns() -> list[re.Pattern[str]]:
    meta_patterns = [
        # Common prefixes - including standalone acknowledgments
        r"^\s*(?:sure|okay|alright|certainly|absolutely)(?:[,!\s]+|$)",
        r"^\s*(?:here\s+is|here\'s)\s+",
        r"^\s*(?:improved|refined|enhanced|better)\s+(?:answer|response|version)",
        r"^\s*(?:let me|i will|i\'ll)\s+(?:provide|give|present)",
        r"^\s*(?:my\s+)?(?:improved|refined|enhanced)\s+(?:answer|response)",
        # Greetings
        r"^\s*(?:hello|hi|hey|greetings)!?\s*,?\s*",
        r"^\s*(?:i am|i\'m)\s+(?:here to )?help",
        # Meta phrases at the start
        r"^\s*(?:as requested|as you asked)",
        r"^\s*(?:below is|following is)",
    ]
    return [re.compile(pattern, re.IGNORECASE) for pattern in meta_patterns]


def _is_meta_commentary_line(
    line: str, patterns: list[re.Pattern[str]]
) -> bool:
    for pattern in patterns:
        if pattern.match(line):
            return True
    return False


def _process_lines_remove_meta(
    lines: list[str], patterns: list[re.Pattern[str]]
) -> tuple[list[str], list[str]]:
    clean_lines = []
    removed_lines = []
    started_content = False

    for line in lines:
        stripped_line = line.strip()

        # Skip empty lines at the beginning
        if not started_content and not stripped_line:
            continue

        # Check if this line is meta-commentary
        is_meta = False
        if not started_content and stripped_line:
            if _is_meta_commentary_line(stripped_line, patterns):
                is_meta = True
                removed_lines.append(stripped_line)

        # If not meta-commentary, it's actual content
        if not is_meta:
            started_content = True
            clean_lines.append(line)

    return clean_lines, removed_lines


def _log_removed_commentary(removed_lines: list[str], logger: Any) -> None:
    if not removed_lines or not logger:
        return

    removed_preview = removed_lines[:3]
    extra = (
        f" (and {len(removed_lines) - 3} more)"
        if len(removed_lines) > 3
        else ""
    )
    logger.debug(
        f"Stripped meta-commentary. Removed: {removed_preview}{extra}"
    )


def _validate_cleaned_text(
    cleaned_text: str, original_text: str, logger: Any | None
) -> str:
    if not cleaned_text and original_text.strip():
        if logger:
            logger.warning(
                "Meta-commentary filter removed all content, returning original"
            )
        return original_text
    return cleaned_text


def strip_meta_commentary(text: str, logger: Any | None = None) -> str:
    if not text or not text.strip():
        return text

    original_text = text
    lines = text.split("\n")

    # Compile patterns and process lines
    patterns = _compile_meta_patterns()
    clean_lines, removed_lines = _process_lines_remove_meta(lines, patterns)

    # Prepare result
    cleaned_text = "\n".join(clean_lines).strip()

    # Log removed commentary
    _log_removed_commentary(removed_lines, logger)

    # Validate and return
    return _validate_cleaned_text(cleaned_text, original_text, logger)

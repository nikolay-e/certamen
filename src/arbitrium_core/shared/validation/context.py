from typing import Any

import litellm

from arbitrium_core.shared.constants import (
    DEFAULT_CONTEXT_SAFETY_MARGIN,
    calculate_safe_max_tokens,
)


def _process_inline_token(token: Any, plain_parts: list[str]) -> None:
    if token.children:
        for child in token.children:
            if child.type == "text":
                plain_parts.append(child.content)
            elif child.type == "code_inline":
                plain_parts.append(child.content)
            elif child.type in ["link_open", "link_close"]:
                continue
    else:
        plain_parts.append(token.content)


def _process_code_block(token: Any, plain_parts: list[str]) -> None:
    plain_parts.append("\n")
    plain_parts.append(token.content)
    plain_parts.append("\n")


def _process_structural_token(token: Any, plain_parts: list[str]) -> None:
    if token.type in [
        "heading_open",
        "heading_close",
        "paragraph_open",
        "paragraph_close",
        "bullet_list_open",
        "bullet_list_close",
        "ordered_list_open",
        "ordered_list_close",
        "softbreak",
        "hardbreak",
    ]:
        plain_parts.append("\n")


def _extract_text_from_tokens(tokens: Any, plain_parts: list[str]) -> None:
    for token in tokens:
        if token.type == "inline":
            _process_inline_token(token, plain_parts)
        elif token.type in ["code_block", "fence"]:
            _process_code_block(token, plain_parts)
        else:
            _process_structural_token(token, plain_parts)

        # Recursively process children
        if token.children:
            _extract_text_from_tokens(token.children, plain_parts)


def markdown_to_plain_text(text: str) -> str:
    import logging
    import re

    from markdown_it import MarkdownIt

    # Disable markdown_it debug logging
    logging.getLogger("markdown_it").setLevel(logging.WARNING)

    # Parse markdown and render to tokens
    md = MarkdownIt()
    tokens = md.parse(text)

    # Extract plain text from tokens
    plain_parts: list[str] = []
    _extract_text_from_tokens(tokens, plain_parts)
    result = "".join(plain_parts)

    # Clean up excessive whitespace
    result = re.sub(r"\n\n+", "\n\n", result)  # Max 2 newlines
    result = re.sub(r" +", " ", result)  # Collapse spaces

    return result.strip()


def estimate_token_count(text: str, model_name: str) -> int:
    count: int = litellm.token_counter(model=model_name, text=text)
    return count


def validate_prompt_size(
    prompt: str,
    model_name: str,
    context_window: int,
    max_tokens: int | None = None,
    safety_margin: float = DEFAULT_CONTEXT_SAFETY_MARGIN,
) -> tuple[bool, int, str]:
    if max_tokens is None:
        max_tokens = calculate_safe_max_tokens(
            context_window, safety_margin=safety_margin
        )

    token_count = estimate_token_count(prompt, model_name)

    # Calculate available space
    safety_tokens = int(context_window * safety_margin)
    available_tokens = context_window - max_tokens - safety_tokens

    if token_count <= available_tokens:
        return (
            True,
            token_count,
            f"Prompt fits within context window ({token_count}/{available_tokens} tokens)",
        )
    else:
        excess_tokens = token_count - available_tokens
        return (
            False,
            token_count,
            (
                f"Prompt exceeds context window by {excess_tokens} tokens ({token_count}/{available_tokens} available)"
            ),
        )

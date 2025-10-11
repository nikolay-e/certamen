#!/usr/bin/env python3
"""Test the meta-commentary filtering functionality."""


from arbitrium.core.comparison import _strip_meta_commentary
from arbitrium.logging import get_contextual_logger, setup_logging

setup_logging(enable_file_logging=False)
logger = get_contextual_logger("test")


def test_remove_sure_here_is_prefix() -> None:
    """Test removing 'Sure, here is' prefix."""
    text = """Sure, here is an improved answer:

This is the actual improved content that should be kept.
It has multiple lines and important information."""

    expected = """This is the actual improved content that should be kept.
It has multiple lines and important information."""

    result = _strip_meta_commentary(text, logger=logger)
    assert result == expected


def test_remove_greeting_prefix() -> None:
    """Test removing greeting prefix."""
    text = """Hello! I am here to help you.

Here's my improved response:
The actual content starts here."""

    expected = "The actual content starts here."
    result = _strip_meta_commentary(text, logger=logger)
    assert result == expected


def test_remove_okay_here_is_prefixes() -> None:
    """Test removing 'Okay,' and 'here is' prefixes."""
    text = """Okay, here is my refined answer:

The core answer content.
More details here."""

    expected = """The core answer content.
More details here."""

    result = _strip_meta_commentary(text, logger=logger)
    assert result == expected


def test_keep_content_without_meta_commentary() -> None:
    """Test keeping content without meta-commentary."""
    text = """This is a direct answer without any meta-commentary.
It just provides the information requested."""

    result = _strip_meta_commentary(text, logger=logger)
    assert result == text


def test_remove_certainly_prefix() -> None:
    """Test removing 'Certainly' prefix."""
    text = """Certainly, I'll provide an improved response:

The main content follows here."""

    expected = "The main content follows here."
    result = _strip_meta_commentary(text, logger=logger)
    assert result == expected


def test_remove_multiple_meta_commentary_lines() -> None:
    """Test removing multiple meta-commentary lines."""
    text = """Sure!
Here's an improved version:
Let me provide you with a better answer:

The actual answer content."""

    expected = "The actual answer content."
    result = _strip_meta_commentary(text, logger=logger)
    assert result == expected


def test_case_insensitive_matching() -> None:
    """Test case insensitive matching."""
    text = """SURE, HERE IS MY IMPROVED ANSWER:

The actual content."""

    expected = "The actual content."
    result = _strip_meta_commentary(text, logger=logger)
    assert result == expected


def test_empty_or_whitespace_only_input() -> None:
    """Test empty or whitespace-only input."""
    text = "   \n\n   "
    result = _strip_meta_commentary(text, logger=logger)
    assert result == text

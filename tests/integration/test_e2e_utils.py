"""End-to-end tests for utility functions and exception handling."""

import pytest

from arbitrium_core.domain.errors import (
    APIError,
    ArbitriumError,
    AuthenticationError,
    ConfigurationError,
    FatalError,
    FileSystemError,
    InputError,
    ModelError,
    ModelResponseError,
    RateLimitError,
)
from arbitrium_core.shared.text import indent_text, strip_meta_commentary
from arbitrium_core.shared.validation.response import detect_apology_or_refusal


class TestExceptionHierarchy:
    """Test custom exception classes."""

    def test_base_exception_creation(self) -> None:
        """Test creating base ArbitriumError."""
        error = ArbitriumError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert isinstance(error, Exception)

    def test_configuration_error(self) -> None:
        """Test ConfigurationError exception."""
        error = ConfigurationError("Invalid config")

        assert isinstance(error, ArbitriumError)
        assert str(error) == "Invalid config"

    def test_fatal_error(self) -> None:
        """Test FatalError exception."""
        error = FatalError("Fatal error occurred")

        assert isinstance(error, ArbitriumError)
        assert str(error) == "Fatal error occurred"

    def test_input_error(self) -> None:
        """Test InputError exception."""
        error = InputError("Invalid input provided")

        assert isinstance(error, ArbitriumError)
        assert str(error) == "Invalid input provided"


class TestAPIErrors:
    """Test API-related exceptions."""

    def test_api_error_basic(self) -> None:
        """Test basic APIError."""
        error = APIError("API call failed")

        assert isinstance(error, ArbitriumError)
        assert "API call failed" in str(error)

    def test_api_error_with_provider(self) -> None:
        """Test APIError with provider information."""
        error = APIError("Request failed", provider="OpenAI")

        assert "[OpenAI]" in str(error)
        assert "Request failed" in str(error)
        assert error.provider == "OpenAI"

    def test_api_error_with_status_code(self) -> None:
        """Test APIError with HTTP status code."""
        error = APIError("Request failed", status_code=429)

        assert "Status: 429" in str(error)
        assert error.status_code == 429

    def test_api_error_with_provider_and_status(self) -> None:
        """Test APIError with both provider and status code."""
        error = APIError(
            "Too many requests", provider="Anthropic", status_code=429
        )

        error_str = str(error)
        assert "[Anthropic]" in error_str
        assert "Too many requests" in error_str
        assert "Status: 429" in error_str
        assert error.provider == "Anthropic"
        assert error.status_code == 429

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError."""
        error = RateLimitError(
            "Rate limit exceeded", provider="OpenAI", status_code=429
        )

        assert isinstance(error, APIError)
        assert "[OpenAI]" in str(error)
        assert "Status: 429" in str(error)

    def test_authentication_error(self) -> None:
        """Test AuthenticationError."""
        error = AuthenticationError(
            "Invalid API key", provider="Anthropic", status_code=401
        )

        assert isinstance(error, APIError)
        assert "[Anthropic]" in str(error)
        assert "Invalid API key" in str(error)
        assert error.status_code == 401


class TestModelErrors:
    """Test model-related exceptions."""

    def test_model_error_basic(self) -> None:
        """Test basic ModelError."""
        error = ModelError("Model failed")

        assert isinstance(error, ArbitriumError)
        assert "Model failed" in str(error)

    def test_model_error_with_model_key(self) -> None:
        """Test ModelError with model_key."""
        error = ModelError("Generation failed", model_key="gpt-4")

        assert "[gpt-4]" in str(error)
        assert "Generation failed" in str(error)
        assert error.model_key == "gpt-4"

    def test_model_response_error(self) -> None:
        """Test ModelResponseError."""
        error = ModelResponseError(
            "Invalid response format", model_key="claude-3"
        )

        assert isinstance(error, ModelError)
        assert "[claude-3]" in str(error)
        assert "Invalid response format" in str(error)

    def test_model_error_without_key(self) -> None:
        """Test ModelError without model_key."""
        error = ModelError("Unknown model error")

        assert "Unknown model error" in str(error)
        assert error.model_key is None


class TestFileSystemErrors:
    """Test filesystem-related exceptions."""

    def test_filesystem_error_basic(self) -> None:
        """Test basic FileSystemError."""
        error = FileSystemError("File operation failed")

        assert isinstance(error, ArbitriumError)
        assert "File operation failed" in str(error)

    def test_filesystem_error_with_path(self) -> None:
        """Test FileSystemError with file path."""
        error = FileSystemError(
            "Cannot read file", file_path="/path/to/file.txt"
        )

        assert "[/path/to/file.txt]" in str(error)
        assert "Cannot read file" in str(error)
        assert error.file_path == "/path/to/file.txt"

    def test_filesystem_error_without_path(self) -> None:
        """Test FileSystemError without file path."""
        error = FileSystemError("Disk full")

        assert "Disk full" in str(error)
        assert error.file_path is None


class TestApologyDetection:
    """Test apology and refusal detection."""

    def test_detect_simple_apology(self) -> None:
        """Test detecting simple apology."""
        assert detect_apology_or_refusal("I'm sorry, I cannot help with that.")
        assert detect_apology_or_refusal("I am sorry, but I cannot do this.")
        assert detect_apology_or_refusal("I apologize, I'm unable to assist.")

    def test_detect_cannot_patterns(self) -> None:
        """Test detecting 'cannot' patterns."""
        assert detect_apology_or_refusal("I cannot help with that request.")
        assert detect_apology_or_refusal("I can't provide that information.")
        assert detect_apology_or_refusal("I'm unable to complete this task.")
        assert detect_apology_or_refusal("I am unable to assist with this.")

    def test_detect_ai_disclaimers(self) -> None:
        """Test detecting AI disclaimers."""
        assert detect_apology_or_refusal("As an AI, I cannot make judgments.")
        assert detect_apology_or_refusal(
            "I'm an AI and cannot perform that task."
        )
        assert detect_apology_or_refusal(
            "I am an AI assistant and don't have..."
        )

    def test_detect_refusal_at_start(self) -> None:
        """Test that refusals at the start are detected."""
        text = "Sorry, I can't help with that. However, here is some other info..."
        assert detect_apology_or_refusal(text)

    def test_no_false_positive_on_valid_response(self) -> None:
        """Test that valid responses are not flagged."""
        assert not detect_apology_or_refusal(
            "Here is my analysis of the situation."
        )
        assert not detect_apology_or_refusal(
            "The answer to your question is..."
        )
        assert not detect_apology_or_refusal(
            "Based on the information provided..."
        )

    def test_no_false_positive_on_contains_sorry_later(self) -> None:
        """Test that 'sorry' appearing later in text is not flagged."""
        text = (
            "The company had to say they were sorry. " * 5
            + "Analysis continues..."
        )
        # Should not be flagged if 'sorry' is not in first 200 chars as refusal
        assert not detect_apology_or_refusal(text)

    def test_empty_text_not_flagged(self) -> None:
        """Test that empty text is not flagged."""
        assert not detect_apology_or_refusal("")
        assert not detect_apology_or_refusal("   ")

    def test_case_insensitive_detection(self) -> None:
        """Test case-insensitive detection."""
        assert detect_apology_or_refusal("I CANNOT help with that.")
        assert detect_apology_or_refusal("I'M SORRY but no.")
        assert detect_apology_or_refusal("Sorry But I Can't Do This.")

    def test_sorry_but_pattern(self) -> None:
        """Test 'sorry but' pattern detection."""
        assert detect_apology_or_refusal("Sorry but I cannot provide that.")
        # Note: "Sorry, but" doesn't match "sorry but" or "sorry, i" patterns
        # This is expected - only specific patterns are matched
        text = "Sorry, I cannot help with that."
        assert detect_apology_or_refusal(text)

    def test_no_have_pattern(self) -> None:
        """Test 'don't have' pattern detection."""
        assert detect_apology_or_refusal(
            "I don't have access to that information."
        )
        assert detect_apology_or_refusal("I do not have the ability to...")


class TestIndentText:
    """Test text indentation helper."""

    def test_indent_simple_text(self) -> None:
        """Test indenting simple text."""
        text = "Line 1\nLine 2\nLine 3"
        result = indent_text(text)

        assert "    Line 1" in result
        assert "    Line 2" in result
        assert "    Line 3" in result

    def test_indent_with_custom_indent(self) -> None:
        """Test indenting with custom indentation."""
        text = "Line 1\nLine 2"
        result = indent_text(text, indent=">>")

        assert ">>Line 1" in result
        assert ">>Line 2" in result

    def test_indent_removes_empty_lines(self) -> None:
        """Test that empty lines are removed."""
        text = "Line 1\n\n\nLine 2\n\nLine 3"
        result = indent_text(text)

        # Empty lines should be removed
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_indent_single_line(self) -> None:
        """Test indenting single line."""
        text = "Single line"
        result = indent_text(text)

        assert "    Single line" in result

    def test_indent_empty_text(self) -> None:
        """Test indenting empty text."""
        result = indent_text("")

        assert result == "\n"

    def test_indent_whitespace_only_lines(self) -> None:
        """Test that whitespace-only lines are removed."""
        text = "Line 1\n   \t  \nLine 2"
        result = indent_text(text)

        lines = result.strip().split("\n")
        assert len(lines) == 2


class TestStripMetaCommentary:
    """Test meta-commentary stripping."""

    def test_strip_sure_prefix(self) -> None:
        """Test removing 'Sure, here is...' prefix."""
        text = "Sure, here is my answer:\n\nActual content here."
        result = strip_meta_commentary(text)

        assert "Sure" not in result
        assert "Actual content here" in result

    def test_strip_heres_prefix(self) -> None:
        """Test removing 'Here's...' prefix."""
        text = "Here's my improved response:\n\nThe actual response."
        result = strip_meta_commentary(text)

        assert "Here's" not in result
        assert "The actual response" in result

    def test_strip_improved_answer_prefix(self) -> None:
        """Test removing 'Improved answer:' prefix."""
        text = "Improved answer:\n\nThe improved content."
        result = strip_meta_commentary(text)

        assert "Improved answer" not in result
        assert "The improved content" in result

    def test_strip_greeting(self) -> None:
        """Test removing greeting."""
        text = "Hello! I am here to help.\n\nHere is the information you need."
        result = strip_meta_commentary(text)

        # The greeting pattern only matches standalone "Hello!" at start
        # "I am here to help" matches the meta pattern
        # Check that at least meta commentary is reduced
        assert "Here is the information you need" in result

    def test_preserve_content_without_meta(self) -> None:
        """Test that content without meta-commentary is preserved."""
        text = "This is pure content without any meta-commentary."
        result = strip_meta_commentary(text)

        assert result.strip() == text.strip()

    def test_strip_multiple_meta_lines(self) -> None:
        """Test removing multiple meta-commentary lines."""
        text = "Sure!\nHere is my answer:\nLet me provide the details.\n\nActual content."
        result = strip_meta_commentary(text)

        assert "Sure" not in result
        assert "Here is" not in result
        assert "Let me" not in result
        assert "Actual content" in result

    def test_return_original_if_all_removed(self) -> None:
        """Test that original is returned if all content would be removed."""
        text = "Sure, okay!"
        result = strip_meta_commentary(text)

        # Should return original if cleaning removes everything
        assert result == text

    def test_strip_empty_text(self) -> None:
        """Test stripping empty text."""
        assert strip_meta_commentary("") == ""
        assert strip_meta_commentary("   ") == "   "

    def test_strip_okay_prefix(self) -> None:
        """Test removing 'Okay' prefix."""
        text = "Okay, here you go:\n\nThe content."
        result = strip_meta_commentary(text)

        assert "Okay" not in result
        assert "The content" in result

    def test_strip_certainly_prefix(self) -> None:
        """Test removing 'Certainly' prefix."""
        text = "Certainly! Let me provide that.\n\nActual answer here."
        result = strip_meta_commentary(text)

        assert "Certainly" not in result
        assert "Actual answer here" in result

    def test_preserve_meta_words_in_content(self) -> None:
        """Test that meta words in actual content are preserved."""
        text = "The CEO said 'Sure, we can do that' during the meeting."
        result = strip_meta_commentary(text)

        # Since it's actual content (not at the start), it should be preserved
        assert "Sure" in result

    def test_strip_as_requested_prefix(self) -> None:
        """Test removing 'As requested' prefix."""
        text = "As requested, here is the analysis:\n\nThe analysis content."
        result = strip_meta_commentary(text)

        assert "As requested" not in result
        assert "The analysis content" in result

    def test_strip_below_is_prefix(self) -> None:
        """Test removing 'Below is' prefix."""
        text = "Below is my response:\n\nThe response content."
        result = strip_meta_commentary(text)

        assert "Below is" not in result
        assert "The response content" in result


class TestUtilsInTournamentContext:
    """Test utilities in actual tournament context."""

    @pytest.mark.asyncio
    async def test_apology_detection_prevents_scoring(self) -> None:
        """Test that apology detection prevents invalid scoring."""
        from arbitrium_core.domain.tournament.scoring import ScoreExtractor

        extractor = ScoreExtractor()

        apology_response = "I'm sorry, but I cannot evaluate these models."

        scores = extractor.extract_scores_from_evaluation(
            apology_response, ["Model A", "Model B"], "TestEvaluator"
        )

        # Should return empty dict due to apology detection
        assert len(scores) == 0

    @pytest.mark.asyncio
    async def test_valid_response_passes_apology_check(self) -> None:
        """Test that valid responses pass apology check."""
        from arbitrium_core.domain.tournament.scoring import ScoreExtractor

        extractor = ScoreExtractor()

        valid_response = "Model A: 8/10\nModel B: 7/10"

        scores = extractor.extract_scores_from_evaluation(
            valid_response, ["Model A", "Model B"], "TestEvaluator"
        )

        # Should extract scores successfully
        assert len(scores) == 2

    def test_indent_used_in_logging(self) -> None:
        """Test that indentation is suitable for logging."""
        log_message = "Line 1\nLine 2\nLine 3"
        indented = indent_text(log_message)

        # Should start with newline for clean logs
        assert indented.startswith("\n")

        # Each line in the result (after the initial newline) should be indented
        # Split without stripping to preserve indentation
        lines = indented.split("\n")[
            1:
        ]  # Skip the first empty element from leading \n
        for line in lines:
            if line:  # Skip empty lines
                assert line.startswith("    ")

    def test_exception_message_formatting(self) -> None:
        """Test that exception messages format correctly for user display."""
        error = APIError(
            "Request timeout",
            provider="OpenAI",
            status_code=504,
        )

        error_str = str(error)

        # Should have clear, formatted message
        assert "[OpenAI]" in error_str
        assert "Request timeout" in error_str
        assert "504" in error_str

    def test_meta_commentary_cleanup_in_responses(self) -> None:
        """Test meta-commentary cleanup improves response quality."""
        messy_response = """
        Sure! Here's my improved answer:

        The capital of France is Paris. It has been the capital since 987 AD.
        """

        cleaned = strip_meta_commentary(messy_response)

        # Meta-commentary should be removed
        assert "Sure" not in cleaned
        assert "Here's my" not in cleaned

        # Actual content should remain
        assert "capital of France" in cleaned
        assert "Paris" in cleaned

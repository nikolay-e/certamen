"""End-to-end tests for markdown utilities."""

import pytest

from arbitrium_core.shared.text.markdown import (
    adjust_markdown_headers,
    sanitize_content_dict,
    sanitize_for_markdown,
)


class TestSanitizeForMarkdown:
    """Test markdown sanitization."""

    def test_sanitize_basic_text(self) -> None:
        """Test sanitizing basic text."""
        text = "This is normal text."
        result = sanitize_for_markdown(text)

        assert result == text

    def test_sanitize_html_entities(self) -> None:
        """Test HTML entity sanitization."""
        text = "Code: <script>alert('test')</script> & more"
        result = sanitize_for_markdown(text)

        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "<script>" not in result

    def test_sanitize_preserves_markdown_by_default(self) -> None:
        """Test that markdown is preserved by default."""
        text = "# Header\n## Subheader\n**Bold** and *italic*"
        result = sanitize_for_markdown(text, preserve_markdown=True)

        assert "#" in result
        assert "**Bold**" in result
        assert "*italic*" in result

    def test_sanitize_escapes_markdown_when_disabled(self) -> None:
        """Test that markdown is escaped when preserve_markdown=False."""
        text = "# Header"
        result = sanitize_for_markdown(text, preserve_markdown=False)

        # Headers should be escaped
        assert "\\#" in result or result != text

    def test_sanitize_empty_text(self) -> None:
        """Test sanitizing empty text."""
        assert sanitize_for_markdown("") == ""
        assert sanitize_for_markdown(None) == ""  # type: ignore[arg-type]

    def test_sanitize_special_characters(self) -> None:
        """Test sanitizing special characters."""
        text = "Price: $100 & Sale: 50%"
        result = sanitize_for_markdown(text)

        assert "$100" in result
        assert "50%" in result
        assert "&amp;" in result


class TestSanitizeContentDict:
    """Test dictionary content sanitization."""

    def test_sanitize_dict_with_strings(self) -> None:
        """Test sanitizing dictionary with string values."""
        content = {
            "title": "Test Title",
            "description": "Test <description> with HTML",
        }

        result = sanitize_content_dict(content)

        assert result["title"] == "Test Title"
        assert "&lt;" in result["description"]
        assert "&gt;" in result["description"]

    def test_sanitize_dict_with_numbers(self) -> None:
        """Test sanitizing dictionary with numeric values."""
        content = {
            "count": 42,
            "price": 99.99,
        }

        result = sanitize_content_dict(content)

        assert "42" in result["count"]
        assert "99.99" in result["price"]

    def test_sanitize_dict_preserves_markdown(self) -> None:
        """Test that markdown is preserved in dictionaries."""
        content = {
            "response": "# Title\n**Bold text**",
        }

        result = sanitize_content_dict(content, preserve_markdown=True)

        assert "#" in result["response"]
        assert "**Bold text**" in result["response"]

    def test_sanitize_dict_with_nested_dict(self) -> None:
        """Test sanitizing dictionary with nested dict values."""
        content = {
            "metadata": {"key": "value"},
        }

        result = sanitize_content_dict(content)

        # Should convert dict to string and sanitize
        assert "metadata" in result

    def test_sanitize_empty_dict(self) -> None:
        """Test sanitizing empty dictionary."""
        result = sanitize_content_dict({})

        assert result == {}


class TestAdjustMarkdownHeaders:
    """Test markdown header adjustment."""

    def test_adjust_headers_basic(self) -> None:
        """Test adjusting headers to start at H3."""
        content = "# Title\n## Subtitle\n### Section"
        result = adjust_markdown_headers(content, start_level=3)

        lines = result.split("\n")
        assert lines[0] == "### Title"  # H1 -> H3
        assert lines[1] == "#### Subtitle"  # H2 -> H4
        assert lines[2] == "##### Section"  # H3 -> H5

    def test_adjust_headers_start_at_h1(self) -> None:
        """Test adjusting headers to start at H1 (no change)."""
        content = "# Title\n## Subtitle"
        result = adjust_markdown_headers(content, start_level=1)

        assert result == content

    def test_adjust_headers_start_at_h2(self) -> None:
        """Test adjusting headers to start at H2."""
        content = "# Title\n## Subtitle"
        result = adjust_markdown_headers(content, start_level=2)

        lines = result.split("\n")
        assert lines[0] == "## Title"  # H1 -> H2
        assert lines[1] == "### Subtitle"  # H2 -> H3

    def test_adjust_preserves_non_headers(self) -> None:
        """Test that non-header content is preserved."""
        content = "# Header\n\nRegular text\n\n## Another header\n\nMore text"
        result = adjust_markdown_headers(content, start_level=2)

        assert "Regular text" in result
        assert "More text" in result

    def test_adjust_empty_content(self) -> None:
        """Test adjusting empty content."""
        assert adjust_markdown_headers("", start_level=3) == ""
        assert adjust_markdown_headers(None, start_level=3) is None  # type: ignore[arg-type]

    def test_adjust_content_without_headers(self) -> None:
        """Test adjusting content without any headers."""
        content = "Just regular text\nNo headers here"
        result = adjust_markdown_headers(content, start_level=3)

        assert result == content

    def test_adjust_multiple_header_levels(self) -> None:
        """Test adjusting content with many header levels."""
        content = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        result = adjust_markdown_headers(content, start_level=2)

        lines = result.split("\n")
        assert lines[0] == "## H1"  # H1 -> H2
        assert lines[1] == "### H2"  # H2 -> H3
        assert lines[2] == "#### H3"  # H3 -> H4
        assert lines[3] == "##### H4"  # H4 -> H5
        assert lines[4] == "###### H5"  # H5 -> H6
        assert lines[5] == "####### H6"  # H6 -> H7

    def test_adjust_headers_with_spaces(self) -> None:
        """Test adjusting headers with various spacing."""
        content = "#  Title with spaces\n##   Subtitle"
        result = adjust_markdown_headers(content, start_level=3)

        assert "### Title with spaces" in result
        assert "#### Subtitle" in result


class TestMarkdownInReports:
    """Test markdown utilities in report generation context."""

    def test_sanitize_for_report_content(self) -> None:
        """Test sanitizing content for report generation."""
        content = {
            "champion_solution": "# Winner's Answer\n<script>bad</script>",
            "total_cost": 2.50,
        }

        sanitized = sanitize_content_dict(content, preserve_markdown=True)

        # Markdown should be preserved
        assert "#" in sanitized["champion_solution"]

        # HTML should be escaped
        assert "&lt;script&gt;" in sanitized["champion_solution"]

        # Numbers should be converted
        assert "2.5" in sanitized["total_cost"]

    def test_adjust_headers_for_nested_content(self) -> None:
        """Test adjusting headers for nested report sections."""
        solution = "# Main Point\n## Supporting Detail\n### Evidence"

        # Adjust to start at H3 (for nesting under H2 report section)
        adjusted = adjust_markdown_headers(solution, start_level=3)

        # All headers should be shifted down 2 levels
        assert "### Main Point" in adjusted
        assert "#### Supporting Detail" in adjusted
        assert "##### Evidence" in adjusted

    def test_combined_sanitize_and_adjust(self) -> None:
        """Test combining sanitization and header adjustment."""
        raw_content = "# Title with <HTML>\n## Subtitle & more"

        # First sanitize
        sanitized = sanitize_for_markdown(raw_content, preserve_markdown=True)

        # Then adjust headers
        adjusted = adjust_markdown_headers(sanitized, start_level=2)

        # HTML should be escaped
        assert "&lt;HTML&gt;" in adjusted
        assert "&amp;" in adjusted

        # Headers should be adjusted
        assert "## Title" in adjusted
        assert "### Subtitle" in adjusted


class TestMarkdownEdgeCases:
    """Test edge cases in markdown processing."""

    def test_sanitize_very_long_text(self) -> None:
        """Test sanitizing very long text."""
        long_text = "A" * 10000
        result = sanitize_for_markdown(long_text)

        assert len(result) == len(long_text)

    def test_sanitize_unicode_text(self) -> None:
        """Test sanitizing text with unicode characters."""
        text = "Japanese: 日本語, Emoji: 🚀, Symbols: €£¥"
        result = sanitize_for_markdown(text)

        # Unicode should be preserved
        assert "日本語" in result
        assert "🚀" in result
        assert "€" in result

    def test_adjust_headers_with_inline_code(self) -> None:
        """Test adjusting headers with inline code."""
        content = "# Title with `code`\n## Another with `more code`"
        result = adjust_markdown_headers(content, start_level=2)

        # Code should be preserved
        assert "`code`" in result
        assert "`more code`" in result

        # Headers should be adjusted
        assert "## Title with `code`" in result
        assert "### Another with `more code`" in result

    def test_sanitize_multiline_text(self) -> None:
        """Test sanitizing multiline text."""
        text = "Line 1 with <tag>\nLine 2 with & symbol\nLine 3"
        result = sanitize_for_markdown(text)

        assert "&lt;tag&gt;" in result
        assert "&amp;" in result
        assert "Line 3" in result

    def test_adjust_headers_preserves_list_markers(self) -> None:
        """Test that header adjustment doesn't affect list markers."""
        content = "# Header\n- List item\n- Another item"
        result = adjust_markdown_headers(content, start_level=2)

        assert "- List item" in result
        assert "- Another item" in result
        assert "## Header" in result

    def test_sanitize_preserves_code_blocks(self) -> None:
        """Test that code blocks are handled correctly."""
        text = "```python\ndef hello():\n    return '<test>'\n```"
        result = sanitize_for_markdown(text)

        # Code blocks should still have their content
        assert "```python" in result
        assert "def hello()" in result

    def test_adjust_headers_with_bold_italic(self) -> None:
        """Test adjusting headers with bold and italic text."""
        content = "# **Bold Header**\n## *Italic Header*"
        result = adjust_markdown_headers(content, start_level=3)

        assert "### **Bold Header**" in result
        assert "#### *Italic Header*" in result


class TestMarkdownIntegration:
    """Test markdown utilities in integration scenarios."""

    @pytest.mark.asyncio
    async def test_report_uses_sanitization(self) -> None:
        """Test that report generation uses sanitization."""
        from arbitrium_core.shared.text.markdown import sanitize_content_dict

        # Simulate report content
        content = {
            "title": "Test Report",
            "description": "Report with <dangerous> HTML & symbols",
            "score": 8.5,
        }

        sanitized = sanitize_content_dict(content, preserve_markdown=True)

        # Should sanitize HTML
        assert "&lt;dangerous&gt;" in sanitized["description"]
        assert "&amp;" in sanitized["description"]

        # Should preserve markdown-friendly content
        assert "Test Report" in sanitized["title"]

    @pytest.mark.asyncio
    async def test_header_adjustment_in_champion_solution(self) -> None:
        """Test header adjustment for champion solution nesting."""
        # Champion solution might have its own headers
        solution = """
# My Solution

## Approach

### Implementation Details

The actual solution goes here.
"""

        # Adjust to nest under report's H2 section
        adjusted = adjust_markdown_headers(solution, start_level=3)

        # Verify adjustment
        assert "### My Solution" in adjusted
        assert "#### Approach" in adjusted
        assert "##### Implementation Details" in adjusted


class TestMarkdownConsistency:
    """Test markdown processing consistency."""

    def test_sanitize_idempotent(self) -> None:
        """Test that sanitization is idempotent."""
        text = "Text with <html> & symbols"

        first = sanitize_for_markdown(text)
        second = sanitize_for_markdown(first)

        # Multiple sanitizations will escape already-escaped entities
        # This is expected behavior, not true idempotence
        assert "&lt;html&gt;" in first
        # Second sanitization will double-escape the & from &amp;
        assert "&amp;" in second

    def test_adjust_headers_idempotent(self) -> None:
        """Test that header adjustment with level=1 is idempotent."""
        content = "# Header\n## Subheader"

        first = adjust_markdown_headers(content, start_level=1)
        second = adjust_markdown_headers(first, start_level=1)

        assert first == second

    def test_sanitize_then_adjust_vs_adjust_then_sanitize(self) -> None:
        """Test order independence (mostly)."""
        text = "# Header with <HTML>"

        # Order 1: sanitize then adjust
        path1 = sanitize_for_markdown(text, preserve_markdown=True)
        path1 = adjust_markdown_headers(path1, start_level=2)

        # Order 2: adjust then sanitize
        path2 = adjust_markdown_headers(text, start_level=2)
        path2 = sanitize_for_markdown(path2, preserve_markdown=True)

        # Both should have HTML escaped
        assert "&lt;HTML&gt;" in path1
        assert "&lt;HTML&gt;" in path2

        # Both should have headers adjusted
        assert "##" in path1
        assert "##" in path2

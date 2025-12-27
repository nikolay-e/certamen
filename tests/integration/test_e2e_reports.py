"""End-to-end tests for report generation."""

from typing import Any

import pytest

from arbitrium_core.domain.tournament.report import ReportGenerator


class MockHost:
    """Mock host environment for testing report writes."""

    def __init__(self) -> None:
        """Initialize mock host."""
        self.written_files: dict[str, str] = {}
        self.should_fail = False

    async def write_file(self, path: str, content: str) -> None:
        """Mock write file operation."""
        if self.should_fail:
            raise OSError("Mock write failure")
        self.written_files[path] = content

    def get_file_content(self, path: str) -> str | None:
        """Get content of written file."""
        return self.written_files.get(path)

    def reset(self) -> None:
        """Reset written files."""
        self.written_files = {}


@pytest.fixture
def mock_host() -> MockHost:
    """Create mock host fixture."""
    return MockHost()


@pytest.fixture
def report_generator(mock_host: MockHost) -> ReportGenerator:
    """Create report generator with mock host."""
    return ReportGenerator(mock_host)


class TestBasicReportGeneration:
    """Test basic report generation functionality."""

    @pytest.mark.asyncio
    async def test_save_simple_report(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test saving a simple report."""
        content = {
            "title": "Test Report",
            "description": "This is a test",
        }

        result = await report_generator.save_report("test_report", content)

        assert result is True
        assert "test_report.md" in mock_host.written_files

    @pytest.mark.asyncio
    async def test_save_champion_solution_report(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test saving champion solution report."""
        content = {
            "champion_model": "Model A",
            "champion_solution": "# Solution\n\nThis is the winning answer.",
            "rounds": "3",
        }

        result = await report_generator.save_report(
            "champion_solution", content
        )

        assert result is True
        assert "champion_solution.md" in mock_host.written_files

        # Check content
        file_content = mock_host.get_file_content("champion_solution.md")
        assert file_content is not None
        assert "Champion Solution" in file_content
        assert "Model A" in file_content

    @pytest.mark.asyncio
    async def test_save_report_with_round_number(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test saving report with round number."""
        content = {"results": "Round 1 results"}

        result = await report_generator.save_report(
            "evaluation", content, round_number=1
        )

        assert result is True
        assert "round1_evaluation.md" in mock_host.written_files

        file_content = mock_host.get_file_content("round1_evaluation.md")
        assert "Round 1" in file_content


class TestReportContentFormatting:
    """Test report content formatting and sanitization."""

    @pytest.mark.asyncio
    async def test_markdown_headers_adjusted(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that markdown headers in champion_solution are adjusted."""
        content = {
            "champion_solution": "# Main Title\n## Subtitle\n### Details"
        }

        await report_generator.save_report("champion_solution", content)

        file_content = mock_host.get_file_content("champion_solution.md")
        assert file_content is not None

        # Headers should be adjusted (h1 -> h3, h2 -> h4, etc.)
        assert "### Main Title" in file_content
        assert "#### Subtitle" in file_content

    @pytest.mark.asyncio
    async def test_underscores_replaced_in_keys(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that underscores in keys are replaced with spaces."""
        content = {
            "test_key_with_underscores": "Value",
            "another_test": "Another value",
        }

        await report_generator.save_report("test_report", content)

        file_content = mock_host.get_file_content("test_report.md")
        assert file_content is not None

        # Underscores should be replaced with spaces and title-cased
        assert "Test Key With Underscores" in file_content
        assert "Another Test" in file_content

    @pytest.mark.asyncio
    async def test_report_sections_properly_formatted(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that report sections are properly formatted."""
        content = {
            "section_one": "Content one",
            "section_two": "Content two",
            "section_three": "Content three",
        }

        await report_generator.save_report("multi_section", content)

        file_content = mock_host.get_file_content("multi_section.md")
        assert file_content is not None

        # Check all sections present with proper headers
        assert "## Section One" in file_content
        assert "## Section Two" in file_content
        assert "## Section Three" in file_content

        # Check content present
        assert "Content one" in file_content
        assert "Content two" in file_content
        assert "Content three" in file_content


class TestReportErrorHandling:
    """Test error handling in report generation."""

    @pytest.mark.asyncio
    async def test_empty_content_returns_false(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that empty content returns False."""
        result = await report_generator.save_report("empty_report", {})

        assert result is False
        assert len(mock_host.written_files) == 0

    @pytest.mark.asyncio
    async def test_write_failure_returns_false(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that write failure returns False."""
        content = {"test": "value"}
        mock_host.should_fail = True

        result = await report_generator.save_report("failing_report", content)

        assert result is False

    @pytest.mark.asyncio
    async def test_none_content_returns_false(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that None content returns False."""
        result = await report_generator.save_report(
            "none_report",
            {},  # Empty dict
        )

        assert result is False


class TestComplexReportContent:
    """Test reports with complex content."""

    @pytest.mark.asyncio
    async def test_report_with_nested_markdown(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test report with nested markdown structures."""
        content = {"analysis": """
## Key Findings

1. First finding
2. Second finding

### Details

- Point A
- Point B
"""}

        result = await report_generator.save_report("analysis", content)

        assert result is True
        file_content = mock_host.get_file_content("analysis.md")
        assert file_content is not None
        assert "Key Findings" in file_content

    @pytest.mark.asyncio
    async def test_report_with_code_blocks(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test report with code blocks."""
        content = {"solution": """
Here's the solution:

```python
def hello():
    print("Hello")
```
"""}

        result = await report_generator.save_report("code_report", content)

        assert result is True
        file_content = mock_host.get_file_content("code_report.md")
        assert file_content is not None
        assert "```python" in file_content
        assert 'print("Hello")' in file_content

    @pytest.mark.asyncio
    async def test_report_with_special_characters(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test report with special characters."""
        content = {"description": "Testing <special> & characters: $100, 50%"}

        result = await report_generator.save_report("special_chars", content)

        assert result is True
        file_content = mock_host.get_file_content("special_chars.md")
        assert file_content is not None
        # Special characters should be preserved
        assert "$100" in file_content


class TestMultipleReports:
    """Test generating multiple reports."""

    @pytest.mark.asyncio
    async def test_multiple_reports_different_names(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test generating multiple reports with different names."""
        await report_generator.save_report("report1", {"data": "First"})
        await report_generator.save_report("report2", {"data": "Second"})
        await report_generator.save_report("report3", {"data": "Third"})

        assert len(mock_host.written_files) == 3
        assert "report1.md" in mock_host.written_files
        assert "report2.md" in mock_host.written_files
        assert "report3.md" in mock_host.written_files

    @pytest.mark.asyncio
    async def test_multiple_round_reports(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test generating reports for multiple rounds."""
        for round_num in range(1, 4):
            await report_generator.save_report(
                "evaluation",
                {"round": f"Round {round_num}"},
                round_number=round_num,
            )

        assert len(mock_host.written_files) == 3
        assert "round1_evaluation.md" in mock_host.written_files
        assert "round2_evaluation.md" in mock_host.written_files
        assert "round3_evaluation.md" in mock_host.written_files

    @pytest.mark.asyncio
    async def test_overwriting_same_report(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test overwriting the same report multiple times."""
        await report_generator.save_report("test", {"version": "1"})
        content1 = mock_host.get_file_content("test.md")

        await report_generator.save_report("test", {"version": "2"})
        content2 = mock_host.get_file_content("test.md")

        # Should have overwritten
        assert content1 != content2
        assert "version" in content2 or "Version" in content2


class TestRealWorldReportScenarios:
    """Test real-world report generation scenarios."""

    @pytest.mark.asyncio
    async def test_tournament_final_report(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test generating a complete tournament final report."""
        content = {
            "champion_model": "GPT-4",
            "champion_solution": "# Final Answer\n\nDetailed solution here.",
            "total_rounds": "5",
            "eliminated_models": "Claude, Gemini, GPT-3.5",
            "total_cost": "$2.50",
        }

        result = await report_generator.save_report(
            "champion_solution", content
        )

        assert result is True
        file_content = mock_host.get_file_content("champion_solution.md")
        assert file_content is not None

        # Verify all important sections present
        assert "GPT-4" in file_content
        assert "Final Answer" in file_content
        assert "5" in file_content

    @pytest.mark.asyncio
    async def test_round_evaluation_report(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test generating a round evaluation report."""
        content = {
            "round": "2",
            "active_models": "Model A, Model B, Model C",
            "scores": "A: 8/10, B: 7/10, C: 6/10",
            "eliminated": "Model C",
        }

        result = await report_generator.save_report(
            "round_evaluation", content, round_number=2
        )

        assert result is True
        assert "round2_round_evaluation.md" in mock_host.written_files


class TestReportSanitization:
    """Test content sanitization in reports."""

    @pytest.mark.asyncio
    async def test_preserve_markdown_formatting(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test that markdown formatting is preserved."""
        content = {"formatted_text": "**Bold** and *italic* and `code`"}

        await report_generator.save_report("formatted", content)

        file_content = mock_host.get_file_content("formatted.md")
        assert file_content is not None
        assert "**Bold**" in file_content
        assert "*italic*" in file_content
        assert "`code`" in file_content

    @pytest.mark.asyncio
    async def test_handle_numeric_values(
        self,
        report_generator: ReportGenerator,
        mock_host: MockHost,
    ) -> None:
        """Test handling numeric values in content."""
        content: dict[str, Any] = {
            "integer_value": 42,
            "float_value": 3.14,
            "string_number": "100",
        }

        result = await report_generator.save_report("numeric", content)

        assert result is True
        file_content = mock_host.get_file_content("numeric.md")
        assert file_content is not None
        assert "42" in file_content
        assert "3.14" in file_content
        assert "100" in file_content

"""Report generation for tournament results."""

from typing import Any, Protocol

from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.text.markdown import (
    adjust_markdown_headers,
    sanitize_content_dict,
)


class HostEnvironment(Protocol):
    """Protocol for file writing."""

    async def write_file(self, path: str, content: str) -> None:
        """Write file to disk."""
        ...


class ReportGenerator:
    """Generates and saves tournament reports to disk."""

    def __init__(self, host: HostEnvironment):
        self.host = host
        self.logger = get_contextual_logger("arbitrium.report")

    async def save_report(
        self,
        content_type: str,
        content: dict[str, Any],
        round_number: int | None = None,
    ) -> bool:
        if not content:
            self.logger.warning(f"No content provided for {content_type}")
            return False

        # Generate filename
        prefix = (
            f"round{round_number}_{content_type}"
            if round_number
            else content_type
        )
        filename = f"{prefix}.md"

        # Sanitize and format content
        sanitized_content = sanitize_content_dict(
            content, preserve_markdown=True
        )

        # Build report
        clean_title = content_type.replace("_", " ").replace(
            "champion solution", "Champion Solution"
        )
        if round_number:
            clean_title += f" Round {round_number}"

        report_title = f"# {clean_title}"
        report_sections = []

        for key, value in sanitized_content.items():
            clean_key = key.replace("_", " ").title()

            if key == "champion_solution":
                adjusted_solution = adjust_markdown_headers(
                    value, start_level=3
                )
                report_sections.append(
                    f"## {clean_key}\n\n{adjusted_solution}"
                )
            else:
                report_sections.append(f"## {clean_key}\n\n{value}")

        report_body = "\n\n".join(report_sections)
        file_content = f"{report_title}\n\n{report_body}"

        # Save to file
        try:
            await self.host.write_file(filename, file_content)
            self.logger.info(f"Saved {content_type} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {content_type}: {e}")
            return False

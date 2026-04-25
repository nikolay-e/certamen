from typing import Any, Protocol

from certamen.shared.logging import get_contextual_logger
from certamen.shared.text.markdown import (
    adjust_markdown_headers,
    sanitize_content_dict,
)


class ReportHostProtocol(Protocol):
    async def write_file(self, path: str, content: str) -> None: ...


class ReportGenerator:
    def __init__(self, host: ReportHostProtocol):
        self.host = host
        self.logger = get_contextual_logger("certamen.report")

    async def save_report(
        self,
        content_type: str,
        content: dict[str, Any],
        round_number: int | None = None,
    ) -> bool:
        if not content:
            self.logger.warning("No content provided for %s", content_type)
            return False

        # Generate filename
        prefix = (
            f"round{round_number}_{content_type}"
            if round_number is not None
            else content_type
        )
        filename = f"{prefix}.md"

        sanitized_content = sanitize_content_dict(
            content, preserve_markdown=True
        )

        clean_title = content_type.replace("_", " ").replace(
            "champion solution", "Champion Solution"
        )
        if round_number is not None:
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
            self.logger.info("Saved %s to %s", content_type, filename)
            return True
        except Exception as e:
            self.logger.error("Failed to save %s: %s", content_type, e)
            return False

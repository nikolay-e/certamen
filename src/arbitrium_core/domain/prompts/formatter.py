"""Dynamic prompt formatting with configurable delimiters for Arbitrium Framework."""

from typing import Any, ClassVar, Literal

DelimiterStyle = Literal["default", "compact", "markdown"]


class DelimiterConfig:
    """Configuration for delimiter styles."""

    STYLES: ClassVar[dict[str, dict[str, Any]]] = {
        "default": {
            "line_char": "=",
            "line_length": 10,
            "template": "{line} {label} {keyword} {line}",
        },
        "compact": {
            "line_char": "-",
            "line_length": 5,
            "template": "{line} {label} {keyword} {line}",
        },
        "markdown": {
            "line_char": "#",
            "line_length": 2,
            "template": "{line} {label} {keyword}",
        },
    }

    @classmethod
    def get_style(cls, style_name: DelimiterStyle) -> dict[str, Any]:
        return cls.STYLES.get(style_name, cls.STYLES["default"])


class PromptFormatter:
    """Formats prompts with dynamic delimiters."""

    def __init__(self, delimiter_style: DelimiterStyle = "default") -> None:
        self.style_config: dict[str, Any] = DelimiterConfig.get_style(
            delimiter_style
        )

    def _make_delimiter(self, label: str, keyword: str) -> str:
        line_char = str(self.style_config["line_char"])
        line_length = int(self.style_config["line_length"])
        template = str(self.style_config["template"])

        line = line_char * line_length
        return template.format(line=line, label=label.upper(), keyword=keyword)

    def wrap_section(
        self, label: str, content: str, strip: bool = True
    ) -> str:
        begin_delimiter = self._make_delimiter(label, "BEGIN")
        end_delimiter = self._make_delimiter(label, "END")

        content_to_wrap = content.strip() if strip else content

        return f"{begin_delimiter}\n\n{content_to_wrap}\n\n{end_delimiter}"

    def wrap_multiple(
        self, sections: list[tuple[str, str]], strip: bool = True
    ) -> str:
        wrapped = [
            self.wrap_section(label, content, strip)
            for label, content in sections
        ]
        return "\n\n".join(wrapped)

    def format_response_wrapper(self, name: str, response: str) -> str:
        label = f"{name} RESPONSE"
        return self.wrap_section(label, response)

    def format_feedback_wrapper(self, reviewer: str, text: str) -> str:
        label = f"FEEDBACK FROM {reviewer}"
        return self.wrap_section(label, text)

    def format_log_message(
        self, message_type: str, model: str, content: str
    ) -> str:
        label = (
            f"[{message_type}] FROM {model}"
            if message_type == "RESPONSE"
            else f"{message_type} TO {model}"
        )
        return self.wrap_section(label, content)

    def format_judge_evaluation(self, judge: str, content: str) -> str:
        label = f"JUDGE EVALUATION FROM {judge}"
        return self.wrap_section(label, content)

    def format_feedback_log(
        self, reviewer: str, target: str, content: str
    ) -> str:
        label = f"FEEDBACK FROM {reviewer} FOR {target}"
        return self.wrap_section(label, content)

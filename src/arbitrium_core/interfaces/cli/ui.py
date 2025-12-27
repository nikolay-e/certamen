import logging

from colorama import Fore

from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.terminal import DEFAULT_COLOR, MODEL_COLORS

logger = get_contextual_logger("arbitrium.interfaces.cli.ui")


class Display:
    def __init__(
        self,
        use_color: bool = True,
        model_colors: dict[str, str] | None = None,
    ):
        self.use_color = use_color and self._should_use_color()
        self.model_colors = model_colors or MODEL_COLORS
        self.default_color = DEFAULT_COLOR

    def _should_use_color(self) -> bool:
        from arbitrium_core.shared.terminal import should_use_color

        return should_use_color()

    def get_color_for_model(self, model_name: str) -> str:
        if not self.use_color:
            return ""
        color: str = self.model_colors.get(model_name, DEFAULT_COLOR)
        return color

    def print(
        self, text: str, level_or_color: str = DEFAULT_COLOR, end: str = "\n"
    ) -> None:
        level_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "success": logging.INFO,
            "debug": logging.DEBUG,
        }

        try:
            clean_text = text.encode("ascii", errors="replace").decode("ascii")

            if level_or_color in level_map:
                log_level = level_map[level_or_color]
                if log_level == logging.DEBUG:
                    logger.debug(clean_text)
                elif log_level == logging.INFO:
                    logger.info(clean_text)
                elif log_level == logging.WARNING:
                    logger.warning(clean_text)
                elif log_level == logging.ERROR:
                    logger.error(clean_text)
                else:
                    logger.info(clean_text)
            else:
                ansi_color = level_or_color if self.use_color else None
                logger.info(
                    clean_text,
                    display_type="colored_text",
                    ansi_color=ansi_color,
                )
        except BrokenPipeError:
            logger.debug(
                "Broken pipe error during print (output may be piped)"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error during print: {e}", exc_info=True
            )

    def print_lines(self, text: str, color: str = DEFAULT_COLOR) -> None:
        lines = text.split("\n")
        for line in lines:
            self.print(line, color)

    def print_model_response(
        self, model_name: str, response_text: str
    ) -> None:
        logger.info(
            response_text,
            display_type="model_response",
            model_name=model_name,
        )

    def print_header(
        self, text: str, char: str = "=", color: str = Fore.CYAN
    ) -> None:
        self.print("\n" + char * 50, color)
        self.print(text, color)
        self.print(char * 50, color)

    def print_section_header(
        self, text: str, color: str = DEFAULT_COLOR
    ) -> None:
        self.print(f"\n--- {text} ---", color)

    def reset(self) -> None:
        pass

    def error(self, text: str) -> None:
        self.print(text, Fore.RED)

    def success(self, text: str) -> None:
        self.print(text, Fore.GREEN)

    def warning(self, text: str) -> None:
        self.print(text, Fore.YELLOW)

    def cyan(self, text: str) -> None:
        self.print(text, Fore.CYAN)

    def info(self, text: str) -> None:
        self.print(text)

    def header(self, text: str) -> None:
        self.print_header(text)


_display = Display()


def configure_display(*, use_color: bool | None = None) -> None:
    global _display  # noqa: PLW0603 - intentional module-level state
    if use_color is None:
        _display = Display()
    else:
        _display = Display(use_color=use_color)


def cli_error(text: str) -> None:
    _display.error(text)


def cli_success(text: str) -> None:
    _display.success(text)


def cli_warning(text: str) -> None:
    _display.warning(text)


def cli_cyan(text: str) -> None:
    _display.cyan(text)


def cli_info(text: str) -> None:
    _display.info(text)

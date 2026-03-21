"""Display utilities for Certamen Framework."""

import logging

from colorama import Fore, Style

from certamen.logging import get_contextual_logger

# Note: colorama is initialized once in main.py

logger = get_contextual_logger("certamen.utils.display")

# Base color palette (excluding red for model outputs)
COLORS = [
    Fore.CYAN,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.WHITE,
    Fore.LIGHTCYAN_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTYELLOW_EX,
    Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTWHITE_EX,
]

# Style variations
STYLES = [
    "",  # No style
    Style.BRIGHT,  # Bright style
]

# Generate a comprehensive color mapping for models
MODEL_COLORS = {}

# Generate colors for numbered models (LLM1, LLM2, etc.)
for i in range(1, 100):  # Support up to 100 models
    color_index = (i - 1) % len(COLORS)
    style_index = ((i - 1) // len(COLORS)) % len(STYLES)
    MODEL_COLORS[f"LLM{i}"] = COLORS[color_index] + STYLES[style_index]

# Generate colors for lettered models (Model A, Model B, etc.)
for i in range(26):  # A-Z
    color_index = i % len(COLORS)
    style_index = (i // len(COLORS)) % len(STYLES)
    MODEL_COLORS[f"Model {chr(65 + i)}"] = (
        COLORS[color_index] + STYLES[style_index]
    )

# Add special system colors
MODEL_COLORS["warning"] = Fore.YELLOW
MODEL_COLORS["error"] = Fore.RED
MODEL_COLORS["success"] = Fore.GREEN
MODEL_COLORS["info"] = Fore.CYAN

# Default color for text
DEFAULT_COLOR = Fore.WHITE


class Display:
    """Handles display formatting and colorization for Certamen Framework."""

    def __init__(
        self,
        use_color: bool = True,
        model_colors: dict[str, str] | None = None,
    ):
        self.use_color = use_color and self._should_use_color()
        self.model_colors = model_colors or MODEL_COLORS
        self.default_color = DEFAULT_COLOR

    def _should_use_color(self) -> bool:
        """Determine if color should be used based on terminal capabilities and environment.

        Returns:
            Boolean indicating whether colors should be used
        """
        # Use the centralized terminal utility function for color detection
        from .terminal import should_use_color

        return should_use_color()

    def get_color_for_model(self, model_name: str) -> str:
        """Returns the color code for a given model name.

        Args:
            model_name: The name of the model

        Returns:
            ANSI color code for the model
        """
        if not self.use_color:
            return ""
        color: str = self.model_colors.get(model_name, DEFAULT_COLOR)
        return color

    def print(
        self, text: str, level_or_color: str = DEFAULT_COLOR, end: str = "\n"
    ) -> None:
        """Prints text through the logging system with proper level and formatting.

        Args:
            text: The text to print
            level_or_color: Log level ("info", "warning", "error", "success") or color code
            end: End string (ignored when using logger)
        """
        # Map level strings to logging levels
        level_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "success": logging.INFO,
            "debug": logging.DEBUG,
        }

        try:
            if level_or_color in level_map:
                log_level = level_map[level_or_color]
                # Clean text for consistent logging (remove emojis for file logs)
                clean_text = text.encode("ascii", errors="replace").decode(
                    "ascii"
                )
                # Use the appropriate method based on log level
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
                # Fallback to direct print for backwards compatibility with color codes
                if self.use_color:
                    print(f"{level_or_color}{text}{Style.RESET_ALL}", end=end)
                else:
                    print(text, end=end)
        except BrokenPipeError:
            # Handle broken pipe error gracefully (e.g., when output is piped to head/less)
            logger.debug(
                "Broken pipe error during print (output may be piped)"
            )
        except Exception as e:
            # Log unexpected print errors
            logger.warning(
                f"Unexpected error during print: {e}", exc_info=True
            )

    def print_lines(self, text: str, color: str = DEFAULT_COLOR) -> None:
        """Prints text in specified color, handling multiple lines.

        Args:
            text: The text to print (can contain newlines)
            color: ANSI color code
        """
        # Split text by newlines to handle multi-line strings
        lines = text.split("\n")
        for line in lines:
            self.print(line, color)

    def print_model_response(
        self, model_name: str, response_text: str
    ) -> None:
        """Prints a model's response with consistent color coding throughout.

        Note: This prints to console only, not to file (to avoid duplicate logging).

        Args:
            model_name: The name of the model
            response_text: The response text from the model
        """
        color = self.get_color_for_model(model_name)

        # Print directly to console to avoid duplicate file logging
        # (responses are already logged to file via logger.info() in comparison.py)
        if self.use_color:
            print(f"\n{color}Model: {model_name}{Style.RESET_ALL}")
            print(f"{color}Response:{Style.RESET_ALL}")
            for line in response_text.split("\n"):
                print(f"{color}{line}{Style.RESET_ALL}")
            print(f"{DEFAULT_COLOR}{'-' * 20}{Style.RESET_ALL}")
        else:
            print(f"\nModel: {model_name}")
            print("Response:")
            print(response_text)
            print("-" * 20)

    def print_header(
        self, text: str, char: str = "=", color: str = Fore.CYAN
    ) -> None:
        """Prints a header with decoration.

        Args:
            text: The header text
            char: Character to use for decoration
            color: ANSI color code
        """
        self.print("\n" + char * 50, color)
        self.print(text, color)
        self.print(char * 50, color)

    def print_section_header(
        self, text: str, color: str = DEFAULT_COLOR
    ) -> None:
        """Prints a section header.

        Args:
            text: The section header text
            color: ANSI color code
        """
        self.print(f"\n--- {text} ---", color)

    def reset(self) -> None:
        """Reset all styling."""
        if self.use_color:
            print(Style.RESET_ALL, end="")

    def error(self, text: str) -> None:
        """Prints an error message in red.

        Args:
            text: The error message to print
        """
        self.print(text, Fore.RED)

    def info(self, text: str) -> None:
        """Prints an informational message.

        Args:
            text: The message to print
        """
        self.print(text)

    def header(self, text: str) -> None:
        """Prints a header message.

        Args:
            text: The header text to print
        """
        self.print_header(text)

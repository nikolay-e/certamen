"""Terminal capability detection utilities for Arbitrium Framework."""

import os
import sys

from colorama import Fore, Style

from arbitrium_core.shared.constants import ANONYMOUS_MODEL_PREFIX

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
MODEL_COLORS: dict[str, str] = {}

# Generate colors for numbered models (LLM1, LLM2, etc.)
for i in range(1, 100):  # Support up to 100 models
    color_index = (i - 1) % len(COLORS)
    style_index = ((i - 1) // len(COLORS)) % len(STYLES)
    MODEL_COLORS[f"{ANONYMOUS_MODEL_PREFIX}{i}"] = (
        COLORS[color_index] + STYLES[style_index]
    )

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


def should_use_color() -> bool:
    # Check for explicit color disable via environment variables
    if os.environ.get("NO_COLOR") is not None:
        return False

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for dumb terminals
    term = os.environ.get("TERM", "").lower()
    if term == "dumb":
        return False

    # Check Windows terminal support
    platform = sys.platform.lower()
    if platform.startswith("win"):
        # Check if we're in Windows Terminal, which supports colors
        # PowerShell, cmd, git bash, and others have this set
        return "WT_SESSION" in os.environ or "ANSICON" in os.environ

    # Most Unix/Linux/Mac terminals support colors
    return True


def strip_ansi_codes(text: str) -> str:
    # This pattern matches ANSI escape codes like \x1b[31m
    import re

    ansi_pattern = re.compile(r"\x1B\[[0-9;]*[mK]")
    return ansi_pattern.sub("", text)

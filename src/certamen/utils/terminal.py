"""Terminal capability detection utilities for Certamen Framework."""

import os
import sys


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

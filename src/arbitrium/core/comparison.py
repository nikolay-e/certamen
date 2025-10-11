"""
Backwards compatibility module for ModelComparison.

DEPRECATED: This module exists for backwards compatibility only.
Use `from arbitrium.core.tournament import Tournament` instead.

The refactoring split the monolithic comparison.py into focused modules:
- tournament.py - Main Tournament class (previously ModelComparison)
- scorer.py - Score extraction and normalization
- anonymizer.py - Model anonymization
- report.py - Report generation
- helpers.py - Utility functions
"""

# Re-export extracted modules for old imports
from .anonymizer import ModelAnonymizer
from .helpers import indent_text as _indent_text
from .helpers import strip_meta_commentary as _strip_meta_commentary
from .report import ReportGenerator
from .scorer import ScoreExtractor

# Re-export everything from tournament for backwards compatibility
from .tournament import ModelComparison  # Main class
from .tournament import EventHandler, HostEnvironment, InitialCosts, TournamentRunner

__all__ = [
    # Supporting classes
    "EventHandler",
    "HostEnvironment",
    "InitialCosts",
    "ModelAnonymizer",
    # Main tournament class
    "ModelComparison",
    "ReportGenerator",
    "ScoreExtractor",
    "TournamentRunner",
    # Helper functions
    "_indent_text",
    "_strip_meta_commentary",
]

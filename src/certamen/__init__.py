"""
Certamen Framework - Multi-LLM Tournament System.

This package provides a framework for comparing and evaluating multiple LLMs
through a tournament-style competition system.

Example:
    >>> from certamen import Certamen
    >>>
    >>> # Initialize from config file
    >>> certamen = await Certamen.from_config("config.yml")
    >>>
    >>> # Run tournament
    >>> result, metrics = await certamen.run_tournament("Your question here")
    >>> print(f"Winner: {metrics['champion_model']}, Cost: ${metrics['total_cost']:.4f}")
    >>>
    >>> # Or run single model
    >>> response = await certamen.run_single_model("gpt-4", "Hello!")
"""

from .__about__ import __version__
from .certamen import Certamen

__all__ = [
    "Certamen",
    "__version__",
]

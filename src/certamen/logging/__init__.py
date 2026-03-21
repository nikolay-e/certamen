"""Logging module for Certamen Framework."""

from .setup import setup_logging
from .structured import get_contextual_logger

__all__ = ["get_contextual_logger", "setup_logging"]

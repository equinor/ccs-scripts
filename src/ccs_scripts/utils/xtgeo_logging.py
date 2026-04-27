"""Utility to suppress specific xtgeo warnings by message content."""

import contextlib
import logging


class _MessageFilter(logging.Filter):
    """Filter to suppress log messages containing specific patterns."""
    
    def __init__(self, message_patterns):
        super().__init__()
        self.message_patterns = message_patterns
    
    def filter(self, record):
        """Return False (suppress) if message contains any pattern."""
        message = record.getMessage()
        return not any(pattern in message for pattern in self.message_patterns)


def suppress_xtgeo_warning_by_message(*message_patterns):
    """Suppress specific xtgeo warnings by message content. Use as context manager.
    
    Args:
        *message_patterns: String patterns to match in warning messages.
                          Warnings containing any of these will be suppressed.
    
    Example:
        with suppress_xtgeo_warning_by_message("Unknown simulator code"):
            init = xtgeo.gridproperties_from_file(...)
    """
    @contextlib.contextmanager
    def _suppress():
        logger = logging.getLogger("xtgeo")
        message_filter = _MessageFilter(message_patterns)
        logger.addFilter(message_filter)
        try:
            yield
        finally:
            logger.removeFilter(message_filter)
    
    return _suppress()

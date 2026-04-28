"""Utility to suppress specific xtgeo warnings by message content."""

import contextlib
import logging


def setup_xtgeo_logging():
    logger = logging.getLogger("xtgeo")
    logger.setLevel(logging.WARNING)


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
        # Get the root xtgeo logger
        xtgeo_logger = logging.getLogger("xtgeo")
        message_filter = _MessageFilter(message_patterns)

        # Add filter to xtgeo logger
        xtgeo_logger.addFilter(message_filter)

        # Add filter to all xtgeo handlers
        handlers = list(xtgeo_logger.handlers)
        for handler in handlers:
            handler.addFilter(message_filter)

        # Also add to root logger handlers in case xtgeo propagates
        root_logger = logging.getLogger()
        root_handlers = list(root_logger.handlers)
        for handler in root_handlers:
            handler.addFilter(message_filter)

        try:
            yield
        finally:
            # Remove filters
            xtgeo_logger.removeFilter(message_filter)
            for handler in handlers:
                handler.removeFilter(message_filter)
            for handler in root_handlers:
                handler.removeFilter(message_filter)

    return _suppress()

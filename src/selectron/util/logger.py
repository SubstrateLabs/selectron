import logging
import sys

_initialized = False


def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance configured with a basic console handler."""
    global _initialized
    if not _initialized:
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set root level low

        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)  # Console level
        root_logger.addHandler(console_handler)

        _initialized = True

    # Get the specific logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Let handlers control the final level
    return logger

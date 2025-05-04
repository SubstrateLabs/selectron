import logging

from rich.logging import RichHandler

from selectron.util.get_app_dir import get_app_dir

_initialized = False
LOG_FILE = get_app_dir() / "selectron.log"

_file_handler = None


def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance configured to log to a file."""
    global _initialized, _file_handler
    # Debug print: Check if function is called and initialization state
    if not _initialized:
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set root level low

        # Create formatter (simple for now, can customize later)
        log_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-5s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        _file_handler = logging.FileHandler(
            LOG_FILE, mode="w", encoding="utf-8"
        )  # Use write mode to clear on start
        _file_handler.setFormatter(log_formatter)
        _file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(_file_handler)

        # stream handler (for terminal visibility, using rich)
        stream_handler = RichHandler(level=logging.DEBUG, rich_tracebacks=True, show_path=False)
        root_logger.addHandler(stream_handler)

        # Set library levels
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("markdown_it").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

        _initialized = True

    # Get the specific logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Let handlers control the final level
    return logger

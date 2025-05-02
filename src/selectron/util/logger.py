import logging

from selectron.util.get_app_dir import get_app_dir

# import sys # No longer needed

_initialized = False
# Define log file path using get_app_dir
LOG_FILE = get_app_dir() / "selectron.log"
# get_app_dir() already ensures the directory exists
# LOG_FILE.parent.mkdir(parents=True, exist_ok=True) # No longer needed here

# Store handler reference
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

        # File handler
        _file_handler = logging.FileHandler(
            LOG_FILE, mode="w", encoding="utf-8"
        )  # Use write mode to clear on start
        _file_handler.setFormatter(log_formatter)
        _file_handler.setLevel(logging.INFO)  # Log INFO and above to file
        root_logger.addHandler(_file_handler)

        # Set library levels (optional, but good practice)
        logging.getLogger("websockets").setLevel(logging.WARNING)

        _initialized = True

    # Get the specific logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Let handlers control the final level
    return logger

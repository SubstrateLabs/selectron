import subprocess
import sys
from pathlib import Path

from selectron.util.logger import get_logger

logger = get_logger(__name__)


def open_log_file(log_path: Path) -> None:
    """Opens the specified log file using the default system application."""
    log_path_str = str(log_path.resolve())
    try:
        if sys.platform == "win32":
            # Use 'start' command which implicitly handles spaces in paths well on Windows
            subprocess.run(["start", "", log_path_str], check=True, shell=True)
        elif sys.platform == "darwin":
            subprocess.run(["open", log_path_str], check=True)
        else:  # Assume Linux/other Unix-like
            subprocess.run(["xdg-open", log_path_str], check=True)
    except FileNotFoundError as e:
        err_msg = f"Error: Could not find command to open log file. Command tried: {e.filename}"
        logger.error(err_msg)
    except subprocess.CalledProcessError as e:
        err_msg = f"Error: Command to open log file failed (code {e.returncode}): {e}"
        logger.error(err_msg)
    except Exception as e:
        err_msg = f"An unexpected error occurred while opening log file: {e}"
        logger.error(err_msg, exc_info=True)

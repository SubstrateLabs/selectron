import sys
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import RichLog

from selectron.util.logger import get_logger
from selectron.util.open_log_file import open_log_file

logger = get_logger(__name__)


class LogPanel(Container):
    def __init__(
        self,
        log_file_path: Path,
        watch_interval: float = 0.5,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._log_file_path = log_file_path
        self._watch_interval = watch_interval
        self._last_log_position = 0
        self._rich_log: Optional[RichLog] = None
        # Ensure log file exists (logger.py should also do this)
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file_path.touch(exist_ok=True)  # Explicitly create if doesn't exist

    @property
    def log_file_path(self) -> Path:
        return self._log_file_path

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield RichLog(highlight=True, markup=True, wrap=True, id="log-panel-internal")

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self._rich_log = self.query_one("#log-panel-internal", RichLog)
        self._clear_log_file()
        self._load_initial_logs()
        self.set_interval(self._watch_interval, self._watch_log_file)

    def _load_initial_logs(self) -> None:
        """Load existing content from the log file into the panel, filtering for INFO+."""
        if not self._rich_log:
            logger.warning("Cannot load initial logs: RichLog not yet available.")
            return
        try:
            if self._log_file_path.exists():
                with open(self._log_file_path, "r", encoding="utf-8") as f:
                    log_content = f.read()
                    self._last_log_position = f.tell()
                    if log_content:
                        self._rich_log.write(log_content)
            else:
                self._rich_log.write(
                    Text(f"Log file not found: {self._log_file_path}", style="yellow")
                )
                self._last_log_position = 0
        except Exception as e:
            err_msg = f"Error loading initial log file {self._log_file_path}: {e}"
            self._rich_log.write(Text(err_msg, style="red"))
            logger.error(err_msg, exc_info=True)

    def _clear_log_file(self) -> None:
        """Truncates the log file."""
        try:
            # Opening with 'w' mode truncates the file.
            open(self._log_file_path, "w", encoding="utf-8").close()
            self._last_log_position = 0  # Reset position after clearing
        except Exception as e:
            err_msg = f"ERROR: Failed to clear log file {self._log_file_path} on mount: {e}"
            # Log to stderr as the logger might write to the file we just failed to clear
            print(err_msg, file=sys.stderr)
            # Also try writing to the panel
            if self._rich_log:
                self._rich_log.write(Text(err_msg + "\n", style="red"))

    async def _watch_log_file(self) -> None:
        """Periodically check the log file for new content, filter for INFO+, and append it."""
        if not self._rich_log:
            return
        try:
            if not self._log_file_path.exists():
                return

            with open(self._log_file_path, "r", encoding="utf-8") as f:
                f.seek(self._last_log_position)
                new_content = f.read()
                if new_content:
                    self._rich_log.write(new_content)
                    self._last_log_position = f.tell()
        except Exception as e:
            # Avoid logging the error back to the log file causing a potential loop
            err_text = Text(f"Error reading log file {self._log_file_path}: {e}\n", style="red")
            self._rich_log.write(err_text)
            # Optionally log to stderr as well
            # print(f"Error reading log file {self._log_file_path}: {e}", file=sys.stderr)

    def open_log_in_editor(self) -> None:
        """Opens the log file using the system's default editor."""
        open_log_file(self._log_file_path)

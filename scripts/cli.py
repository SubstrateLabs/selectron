import asyncio
import logging
import subprocess  # Added for opening file
import sys  # Added back for stderr check
from pathlib import Path  # Added for file watching

# import sys -> Removed as unused
from typing import Optional

import websockets
from PIL import Image
from textual import on  # Import 'on' decorator
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog
from textual.worker import Worker  # Import Worker type

from selectron.chrome.cdp_executor import CdpBrowserExecutor
from selectron.chrome.chrome_cdp import (
    ChromeTab,
    capture_tab_screenshot,
    get_final_url_and_title,
    get_html_via_ws,
    wait_for_page_load,
)
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent
from selectron.chrome.connect import ensure_chrome_connection
from selectron.chrome.types import TabReference
from selectron.dom.dom_attributes import DOM_STRING_INCLUDE_ATTRIBUTES
from selectron.dom.dom_service import DomService
from selectron.util.extract_metadata import HtmlMetadata, extract_metadata
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger
from selectron.util.sample_save import save_sample_data

# --- Logging Setup ---
logger = get_logger(__name__)
# Set library levels (consider moving this to get_logger if applicable globally)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)  # Adjusted level

# Define log file path using get_app_dir (ensure consistency with logger.py)
LOG_FILE = get_app_dir() / "selectron.log"

# --- Textual App ---


class CliApp(App[None]):
    """A Textual app with Chrome monitoring and log file watching."""

    CSS_PATH = "cli.tcss"  # Add path to the CSS file

    BINDINGS = [
        Binding(key="ctrl+c", action="quit", description="Quit App", show=False),
        Binding(key="ctrl+q", action="quit", description="Quit App", show=True),
        Binding(key="ctrl+l", action="open_log_file", description="Open Logs", show=True),
        # Binding(key="ctrl+t", action="toggle_dark", description="Toggle dark mode"), # Example if needed
    ]

    # Initialize dark mode state
    dark = False
    monitor: Optional[ChromeMonitor] = None
    monitor_task: Optional[Worker[None]] = None
    shutdown_event: asyncio.Event

    # Attributes for log file watching
    _log_file_path: Path
    _last_log_position: int
    _active_tab_ref: Optional[TabReference] = None

    def __init__(self):
        super().__init__()
        self.shutdown_event = asyncio.Event()
        # Initialize log watching attributes
        self._log_file_path = LOG_FILE
        self._last_log_position = 0
        # Ensure log file exists (logger.py should also do this)
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file_path.touch(exist_ok=True)  # Explicitly create if doesn't exist

    @property
    def log_panel(self) -> RichLog:
        """Convenience property to access the RichLog widget."""
        return self.query_one("#log-panel", RichLog)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container():  # Main container
            yield Vertical(
                RichLog(highlight=True, markup=False, id="log-panel", wrap=True),
                classes="log-container",
            )
            # Input and Submit button row
            # with Horizontal(classes="input-bar"):  # Container for input and submit - Removed container
            yield Input(placeholder="Enter text to log...", id="log-input")
            # yield Button("Open logs", id="open-log")  # Moved and renamed Open Log button - Removed button
            # Button sits below logs and input
            # yield Horizontal( # Removed this container
        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    async def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Manually clear log file at the start of mounting to ensure clean slate
        try:
            # Opening with 'w' mode truncates the file.
            open(self._log_file_path, "w").close()  # Open in write mode and immediately close.
            # with open(self._log_file_path, "w", encoding="utf-8") as f:
            #     pass # Ensure the file is opened and closed in 'w' mode.
            self._last_log_position = 0  # Reset position after clearing
        except Exception as e:
            # Log error to stderr as logger might not be fully ready or file is problematic
            print(
                f"ERROR: Failed to clear log file {self._log_file_path} on mount: {e}",
                file=sys.stderr,
            )

        # Ensure logger is initialized (though clearing is now manual)
        get_logger(__name__)  # Still good practice to get the logger instance

        # Load initial logs from file (which should now be reliably empty)
        self._load_initial_logs()
        # Start watching the log file
        self.set_interval(0.5, self._watch_log_file)  # Check every 500ms

        # Autofocus the input field
        self.query_one("#log-input", Input).focus()

        logger.info("App mounted. Starting Chrome monitor...")  # This goes to file
        # Run the monitor setup and main loop in the background
        self.monitor_task = self.run_worker(self.run_monitoring(), exclusive=True)

    # --- Log File Watching Methods ---

    def _is_info_or_higher(self, log_line: str) -> bool:
        """Check if a log line string appears to be INFO level or higher."""
        # Simple check based on standard format: YYYY-MM-DD HH:MM:SS [LEVEL] ...
        # Look for the level marker after the timestamp.
        if (
            "[INFO ]" in log_line
            or "[WARN ]" in log_line
            or "[ERROR]" in log_line
            or "[CRIT ]" in log_line
        ):
            return True
        # Also consider lines that might not have the standard prefix (e.g., tracebacks)
        # For now, let's assume non-prefixed lines might be important (like parts of tracebacks)
        # or are INFO level if they don't match DEBUG
        if "[DEBUG]" not in log_line:
            return True  # Show if not explicitly DEBUG
        return False

    def _load_initial_logs(self) -> None:
        """Load existing content from the log file into the panel, filtering for INFO+."""
        try:
            log_panel = self.query_one("#log-panel", RichLog)
            if self._log_file_path.exists():
                with open(self._log_file_path, "r", encoding="utf-8") as f:
                    # Read all content first
                    log_content = f.read()
                    # Get position AFTER reading
                    self._last_log_position = f.tell()

                    # Now filter the lines from the read content
                    lines_to_write = []
                    for line in log_content.splitlines(keepends=True):
                        if self._is_info_or_higher(line):
                            lines_to_write.append(line)
                    log_panel.write("".join(lines_to_write))
            else:
                log_panel.write(f"Log file not found: {self._log_file_path}")
                self._last_log_position = 0
        except Exception as e:
            err_msg = f"Error loading initial log file {self._log_file_path}: {e}"
            try:
                self.query_one("#log-panel", RichLog).write(f"[red]{err_msg}[/red]")
            except Exception:
                pass
            logger.error(err_msg, exc_info=True)  # Log error to file

    def _watch_log_file(self) -> None:
        """Periodically check the log file for new content, filter for INFO+, and append it."""
        try:
            if not self._log_file_path.exists():
                return

            log_panel = self.query_one("#log-panel", RichLog)
            with open(self._log_file_path, "r", encoding="utf-8") as f:
                f.seek(self._last_log_position)
                new_content = f.read()
                if new_content:
                    # Filter new lines before writing
                    lines_to_write = []
                    # Split potentially multiple new lines, filter, then join
                    for line in new_content.splitlines(keepends=True):
                        if self._is_info_or_higher(line):
                            lines_to_write.append(line)
                    if lines_to_write:
                        log_panel.write("".join(lines_to_write))
                    # Update position regardless of filtering
                    self._last_log_position = f.tell()
        except Exception as e:
            logger.error(
                f"Error reading log file {self._log_file_path}: {e}", exc_info=True
            )  # Log error to file

    # --- Input Handling ---

    @on(Input.Submitted, "#log-input")  # Handle Enter key on the specific input
    def handle_log_input_submission(self, event: Input.Submitted) -> None:
        """Logs the input field's content when Enter is pressed."""
        if event.value:  # Only log if there's text
            logger.info(f"User Input Logged: {event.value}")
            event.input.clear()  # Clear the input after logging
            self.run_worker(self._highlight_active_tab_body(), exclusive=False)
        else:
            logger.debug(
                "Log input submitted but was empty."
            )  # Optional: log empty submissions as debug

    # --- Monitoring ---

    async def run_monitoring(self) -> None:
        """Runs the main Chrome monitoring loop."""
        if not await ensure_chrome_connection():
            logger.error("Failed to establish Chrome connection.")
            return
        self.monitor = ChromeMonitor(check_interval=1.5)
        try:
            logger.info("Initializing Chrome monitor...")
            monitor_started = await self.monitor.start_monitoring(
                on_polling_change_callback=self._handle_polling_change,
                on_interaction_update_callback=self._handle_interaction_update,
                on_content_fetched_callback=self._handle_content_fetched,
            )
            if not monitor_started:
                logger.error("Failed to start monitor.")
                return
            logger.info("Monitor started successfully.")
            await self.shutdown_event.wait()
            logger.info("Shutdown signal received.")
        except Exception as e:
            logger.exception(f"Error in monitoring loop: {e}")
            if not self.shutdown_event.is_set():
                self.shutdown_event.set()
        finally:
            logger.info("Monitor worker shutting down...")
            if self.monitor:
                logger.info("Stopping Chrome monitor...")
                await self.monitor.stop_monitoring()
                logger.info("Monitor stopped.")
            else:
                logger.warning("Monitor object not initialized, skipping stop.")
            logger.info("Monitor worker finished.")
            self.app.exit()

    async def _handle_polling_change(self, event: TabChangeEvent):
        """Callback function for tab changes detected ONLY by polling."""
        tasks = []
        for new_tab in event.new_tabs:
            if new_tab.webSocketDebuggerUrl:
                logger.info(f"Polling: New Tab: {new_tab.title} ({new_tab.url})")
                tasks.append(self._process_new_tab(new_tab))
            else:
                logger.warning(f"Polling: New tab {new_tab.id} missing websocket URL.")

        for closed_ref in event.closed_tabs:
            logger.info(f"Polling: Closed Tab: ID {closed_ref.id} ({closed_ref.url})")

        for navigated_tab, old_ref in event.navigated_tabs:
            logger.info(
                f"Polling: Navigated Tab: ID {navigated_tab.id} from {old_ref.url} TO {navigated_tab.url}"
            )
            if navigated_tab.webSocketDebuggerUrl:
                tasks.append(self._process_new_tab(navigated_tab))
            else:
                logger.warning(f"Polling: Navigated tab {navigated_tab.id} missing websocket URL.")

        if tasks:
            # Run these tasks concurrently but manage them within the worker
            await asyncio.gather(*tasks)

    async def _handle_interaction_update(self, ref: TabReference):
        logger.debug(f"Interaction Update: Tab {ref.id} ({ref.url})")
        pass  # Placeholder

    async def _handle_content_fetched(
        self,
        ref: TabReference,
        image: Optional[Image.Image],
        scroll_y: Optional[int],
        dom_string: Optional[str],
    ):
        """Callback triggered after interaction + debounce + content fetch."""
        logger.info(f"Interaction: Content Fetched: Tab {ref.id} ({ref.url}) ScrollY: {scroll_y}")
        self._active_tab_ref = ref

        # Log if DOM string was fetched
        if dom_string:
            logger.info(f"    Fetched DOM (Length: {len(dom_string)})")
        else:
            logger.warning(f"    DOM string not fetched for {ref.url} after interaction.")

        if not ref.html:
            logger.warning(f"    HTML not fetched for {ref.url}, skipping metadata/save.")

        metadata: Optional[HtmlMetadata] = None
        if ref.html:  # Only try if HTML exists
            try:
                metadata = extract_metadata(ref.html, url=ref.url)
            except Exception as e:
                logger.exception(f"Error extracting metadata for {ref.url}: {e}")
        elif not ref.html:
            # Logged above, no need to repeat unless we want more detail
            pass

        if not image:
            logger.warning(f"    No screenshot captured for {ref.url}.")

        # Attempt save if we have the core components (URL, HTML, Metadata)
        if ref.url and ref.html and metadata:
            try:
                # Run save in executor to avoid blocking worker thread? For now, keep inline.
                await save_sample_data(
                    url=ref.url,
                    html=ref.html,
                    metadata=metadata,
                    image=image,  # Can be None
                    dom_string=dom_string,  # Can be None
                )
                logger.info(f"    Sample data saved for {ref.url}")
            except Exception as save_e:
                logger.exception(f"Failed to save sample data for {ref.url}: {save_e}")

        else:
            missing = [
                item
                for item, val in {
                    "Ref URL": ref.url,
                    "HTML": ref.html,
                    "Metadata": metadata,
                }.items()
                if not val
            ]
            logger.warning(f"Skipping save for {ref.url} due to missing: {', '.join(missing)}")

    async def _process_new_tab(self, tab: ChromeTab):
        """Fetches HTML, metadata, screenshot for a NEW or NAVIGATED tab and saves samples. (Instance Method)"""
        html = metadata = screenshot_pil_image = ws = dom_string = None
        final_url = final_title = None
        if not tab.webSocketDebuggerUrl:
            logger.warning(f"Tab {tab.id} missing ws url")
            return
        ws_url = tab.webSocketDebuggerUrl
        try:
            logger.debug(f"Connecting ws: {ws_url}")
            ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024, open_timeout=10)
            logger.debug(f"Connected ws for {tab.id}")
            loaded = await wait_for_page_load(ws)
            logger.debug(f"Page load status {tab.id}: {loaded}")
            await asyncio.sleep(1.0)  # Settle delay
            final_url, final_title = await get_final_url_and_title(
                ws, tab.url, tab.title or "Unknown"
            )
            if final_url:
                html = await get_html_via_ws(ws, final_url)
                if html and final_title:
                    metadata = extract_metadata(html, url=final_url)
                if html:  # Fetch DOM only if HTML exists
                    try:
                        browser_executor = CdpBrowserExecutor(ws_url, final_url, ws_connection=ws)
                        dom_service = DomService(browser_executor)
                        dom_state = await dom_service.get_elements()
                        if dom_state and dom_state.element_tree:
                            dom_string = dom_state.element_tree.elements_to_string(
                                include_attributes=DOM_STRING_INCLUDE_ATTRIBUTES
                            )
                    except Exception as dom_e:
                        logger.exception(f"Error fetching DOM for {final_url}: {dom_e}")
                if final_title:
                    screenshot_pil_image = await capture_tab_screenshot(
                        ws_url=ws_url, ws_connection=ws
                    )
            else:
                logger.warning(f"Could not get final URL for {tab.id}")
        except Exception as e:
            logger.exception(f"Error processing tab {tab.id} ({tab.url}): {e}")
        finally:
            await ws.close() if ws else None
        # --- Summary Log ---
        fetched = [
            name
            for name, val in {
                "HTML": html,
                "DOM": dom_string,
                "SS": screenshot_pil_image,
                "Meta": metadata,
            }.items()
            if val
        ]
        log_url = final_url or tab.url
        log_title = final_title or tab.title
        if final_url:
            # Keep color logic here as it's conditional, but remove tags
            status = "Success" if html and metadata else "Partial"
            logger.info(
                f"Process Result [{status}]: Tab {tab.id} ({log_title} - {log_url}) Fetched: {', '.join(fetched) or 'None'}"
            )
        else:
            logger.error(f"Process Failed: Tab {tab.id} ({tab.url}) - No final URL")
        # --- Save Data ---
        if html and metadata and final_url:
            try:
                await save_sample_data(
                    url=final_url,
                    html=html,
                    metadata=metadata,
                    image=screenshot_pil_image,
                    dom_string=dom_string,
                )
                logger.info(f"Saved {final_url}")
            except Exception as e:
                logger.exception(f"Save failed for {final_url}: {e}")
        elif final_url:
            logger.warning(f"Skipping save for {final_url} (missing HTML/Metadata)")

    async def action_quit(self) -> None:
        """Action to quit the app."""
        logger.info("Shutdown requested...")
        self.shutdown_event.set()
        if self.monitor_task:
            try:
                logger.info("Waiting for monitor worker...")
                await self.monitor_task.wait()
                logger.info("Monitor worker finished.")
            except Exception as e:
                logger.error(f"Error during monitor shutdown wait: {e}")
                if (
                    self.monitor_task
                    and not self.monitor_task.is_cancelled
                    and not self.monitor_task.is_finished
                ):
                    try:
                        logger.warning("Cancelling hung worker...")
                        self.monitor_task.cancel()
                    except Exception as cancel_e:
                        logger.error(f"Error cancelling worker: {cancel_e}")
        logger.info("Exiting application.")
        await asyncio.sleep(0.1)
        self.app.exit()

    # --- Button Action ---

    # def on_button_pressed(self, event: Button.Pressed) -> None: # Removed method
    #     """Handle button clicks.""" # Removed method
    #     if event.button.id == "open-log": # Removed method
    #         self.action_open_log_file() # Removed method
    #     # elif event.button.id == "log-submit": # Removed handler for log-submit # Removed method
    #     #     log_input_widget = self.query_one("#log-input", Input) # Removed method
    #     #     if log_input_widget.value:  # Only log if there's text # Removed method
    #     #         logger.info(f"User Input Logged (Button): {log_input_widget.value}") # Removed method
    #     #         log_input_widget.clear()  # Clear the input after logging # Removed method
    #     #     else: # Removed method
    #     #          logger.debug("Log Submit button pressed but input was empty.")  # Optional # Removed method

    def action_open_log_file(self) -> None:
        """Opens the log file using the default system application."""
        log_path_str = str(self._log_file_path.resolve())
        logger.info(f"Attempting to open log file: {log_path_str}")
        try:
            if sys.platform == "win32":
                # os.startfile(log_path_str) # Alternative for windows
                subprocess.run(["start", "", log_path_str], check=True, shell=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", log_path_str], check=True)
            else:  # Assume Linux/other Unix-like
                subprocess.run(["xdg-open", log_path_str], check=True)
            logger.info("Successfully launched command to open log file.")
        except FileNotFoundError as e:
            # Handle case where 'open', 'xdg-open', or 'start' isn't found
            err_msg = f"Error: Could not find command to open log file. Command tried: {e.filename}"
            logger.error(err_msg)
            # self.notify(err_msg, title="Error Opening Log", severity="error") # Optional TUI notification
        except subprocess.CalledProcessError as e:
            err_msg = f"Error: Command to open log file failed (code {e.returncode}): {e}"
            logger.error(err_msg)
            # self.notify(err_msg, title="Error Opening Log", severity="error")
        except Exception as e:
            err_msg = f"An unexpected error occurred while opening log file: {e}"
            logger.error(err_msg, exc_info=True)
            # self.notify(err_msg, title="Error Opening Log", severity="error")

    async def _highlight_active_tab_body(self) -> None:
        """Highlights the body of the currently tracked active tab."""
        if not self._active_tab_ref or not self._active_tab_ref.ws_url:
            logger.warning("Cannot highlight: No active tab reference or websocket URL.")
            return

        ws_url = self._active_tab_ref.ws_url
        url = self._active_tab_ref.url
        tab_id = self._active_tab_ref.id
        logger.info(f"Attempting to highlight body of tab {tab_id} ({url}) via {ws_url}")

        js_code = """
        (function() {
            const body = document.body;
            if (!body) return 'No body element found.';
            const originalStyle = body.style.outline;
            body.style.outline = '3px solid red';
            setTimeout(() => {
                // Check if style is still ours before removing
                if (body.style.outline === '3px solid red') {
                    body.style.outline = originalStyle;
                }
            }, 1000); // Remove after 1 second
            return 'Body highlighted red temporarily.';
        })();
        """

        try:
            # Use CdpBrowserExecutor to handle connection and execution
            # No need for async with if we let it manage the connection
            executor = CdpBrowserExecutor(ws_url, url)
            result = await executor.evaluate(js_code)
            logger.info(f"Highlight JS execution result for tab {tab_id}: {result}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"Highlight failed for tab {tab_id}: WebSocket error - {e}")
        except Exception as e:
            logger.error(f"Highlight failed for tab {tab_id}: Unexpected error - {e}", exc_info=True)


if __name__ == "__main__":
    # Ensure logger is initialized (and file created) before app runs
    get_logger("__main__")  # Initial call to setup file logging
    app = CliApp()
    app.run()

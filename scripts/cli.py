import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import websockets
from PIL import Image
from pydantic_ai import Agent, Tool
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog
from textual.worker import Worker

from selectron.ai.selector_agent import (
    _DOM_CONTEXT_PROMPT_SECTION,
    _SYSTEM_PROMPT_BASE,
)
from selectron.ai.selector_tools import SelectorEvaluationResult, SelectorTools
from selectron.ai.selector_types import (
    SelectorProposal,
)
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
    _active_tab_dom_string: Optional[str] = None
    _agent_worker: Optional[Worker[None]] = None
    _highlights_active: bool = False  # Track if highlights are currently shown
    _last_highlight_selector: Optional[str] = None  # Remember last selector
    _last_highlight_color: Optional[str] = None  # Remember last color

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
            selector_to_try = event.value
            logger.info(f"User Input (Selector): {selector_to_try}")
            event.input.clear()  # Clear the input after logging

            # --- Cancel previous agent, clear state, and clear highlights ---
            if self._agent_worker and self._agent_worker.is_running:
                logger.debug("Cancelling previous agent worker.")
                self._agent_worker.cancel()
            # Clear highlight state
            self._highlights_active = False
            self._last_highlight_selector = None
            self._last_highlight_color = None
            # Use call_later to ensure visual highlight clearing happens in the event loop
            self.call_later(self._clear_all_highlights)
            # --- Start new agent worker ---
            logger.debug(f"Starting new agent worker for '{selector_to_try}'.")
            self._agent_worker = self.run_worker(
                self._run_agent_and_highlight(selector_to_try), exclusive=False
            )  # Assign the new worker
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
        self.monitor = ChromeMonitor(
            rehighlight_callback=self.trigger_rehighlight, check_interval=1.5
        )
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

        # --- Update Active Tab Reference ---
        # Set the active tab reference ONLY if we successfully got the core data needed for the agent
        if final_url and html and ws_url:
            logger.info(f"Setting active tab reference: {tab.id} ({final_url})")
            self._active_tab_ref = TabReference(
                id=tab.id, url=final_url, title=final_title, html=html, ws_url=ws_url
            )
            self._active_tab_dom_string = dom_string  # Store the DOM string we fetched
        elif self._active_tab_ref and self._active_tab_ref.id == tab.id:
            # If this tab *was* the active one but processing failed, clear it
            logger.warning(f"Clearing active tab reference for {tab.id} due to processing failure.")
            self._active_tab_ref = None
            self._active_tab_dom_string = None

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
                        logger.warning("Cancelling hung monitor worker...")
                        self.monitor_task.cancel()
                    except Exception as cancel_e:
                        logger.error(f"Error cancelling monitor worker: {cancel_e}")

        # --- Cancel agent worker, clear state, and clear highlights on exit --- #
        if self._agent_worker and self._agent_worker.is_running:
            logger.info("Cancelling agent worker on exit...")
            self._agent_worker.cancel()
        # Clear highlight state
        self._highlights_active = False
        self._last_highlight_selector = None
        self._last_highlight_color = None
        # Clear highlights before exiting (fire and forget)
        self.call_later(self._clear_all_highlights)
        await asyncio.sleep(0.1)  # Short delay to allow cleanup task to run
        # --- End cleanup --- #

        logger.info("Exiting application.")
        self.app.exit()

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

    async def _run_agent_and_highlight(self, target_description: str) -> None:
        """Runs the SelectorAgent and highlights the final proposed selector."""
        if not self._active_tab_ref or not self._active_tab_ref.ws_url:
            logger.warning("Cannot run agent: No active tab reference with ws_url.")
            return

        tab_ref = self._active_tab_ref
        ws_url = tab_ref.ws_url  # We know this exists now
        current_html = tab_ref.html
        current_dom_string = self._active_tab_dom_string
        current_url = tab_ref.url

        # --- Fetch data if missing --- #
        if not current_html:
            logger.warning(f"HTML missing for tab {tab_ref.id}. Attempting re-fetch...")
            ws = None
            try:
                # <<< Check ws_url before connecting >>>
                if not ws_url:
                    logger.error(f"Cannot re-fetch for tab {tab_ref.id}: WebSocket URL is missing.")
                    return

                # Establish connection
                ws = await websockets.connect(
                    ws_url, max_size=30 * 1024 * 1024, open_timeout=10, close_timeout=10
                )
                # Wait for load (best effort)
                await wait_for_page_load(ws)
                # Get latest URL/Title (title not used here but good practice)
                latest_url, _ = await get_final_url_and_title(
                    ws, tab_ref.url or "", tab_ref.title or "", tab_ref.id
                )
                current_url = latest_url  # Update URL

                # Fetch HTML
                fetched_html = await get_html_via_ws(ws, latest_url)
                if not fetched_html:
                    logger.error(
                        f"Failed to re-fetch HTML for tab {tab_ref.id}. Aborting agent run."
                    )
                    await ws.close()
                    return
                current_html = fetched_html
                # <<< Remove attempt to modify tab_ref.html >>>
                # tab_ref.html = fetched_html

                # Fetch DOM String using the same connection
                try:
                    # <<< Check latest_url before creating executor >>>
                    if not latest_url:
                        logger.error(
                            f"Cannot fetch DOM for tab {tab_ref.id}: Latest URL is missing after fetch."
                        )
                        current_dom_string = None  # Ensure it's None
                    else:
                        executor = CdpBrowserExecutor(ws_url, latest_url, ws_connection=ws)
                        dom_service = DomService(executor)
                        dom_state = await dom_service.get_elements()
                        if dom_state and dom_state.element_tree:
                            current_dom_string = dom_state.element_tree.elements_to_string(
                                include_attributes=DOM_STRING_INCLUDE_ATTRIBUTES
                            )
                            self._active_tab_dom_string = current_dom_string  # Update stored DOM
                            logger.info(f"Successfully re-fetched DOM string for tab {tab_ref.id}.")
                        else:
                            logger.warning(f"Failed to re-fetch DOM string for tab {tab_ref.id}.")
                            current_dom_string = None  # Ensure it's None if fetch fails
                except Exception as dom_e:
                    logger.error(
                        f"Error re-fetching DOM string for tab {tab_ref.id}: {dom_e}", exc_info=True
                    )
                    current_dom_string = None

                logger.info(f"Successfully re-fetched HTML for tab {tab_ref.id}.")

            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error during re-fetch for tab {tab_ref.id}: {e}")
                return  # Cannot proceed without connection
            except Exception as e:
                logger.error(
                    f"Unexpected error during re-fetch for tab {tab_ref.id}: {e}", exc_info=True
                )
                return  # Cannot proceed
            finally:
                if ws and ws.state != websockets.protocol.State.CLOSED:
                    await ws.close()
        # --- End Fetch data --- #

        # Now we should have HTML, proceed with agent logic
        logger.info(
            f"Running SelectorAgent logic for target '{target_description}' on tab {tab_ref.id}"
        )

        # If URL was updated during fetch, use the latest one
        base_url_for_agent = current_url  # Use the potentially updated current_url
        if not base_url_for_agent:
            logger.error(f"Cannot run agent: Base URL is missing for tab {tab_ref.id}")
            return

        # Check HTML again just to be absolutely sure after fetch block
        if not current_html:
            logger.error(
                f"Cannot run agent: HTML content is missing even after fetch attempt for tab {tab_ref.id}."
            )
            return

        try:
            # --- Setup Tools and Wrappers --- #
            # Use the potentially updated html and url
            tools_instance = SelectorTools(html_content=current_html, base_url=base_url_for_agent)

            async def evaluate_selector_wrapper(selector: str, target_text_to_check: str, **kwargs):
                # Run the actual tool
                logger.debug(f"Agent calling evaluate_selector: '{selector}'")
                result = await tools_instance.evaluate_selector(
                    selector=selector, target_text_to_check=target_text_to_check, **kwargs
                )
                if result and result.element_count > 0 and not result.error:
                    logger.debug(f"Highlighting intermediate selector: '{selector}'")
                    asyncio.create_task(
                        self._highlight_elements_by_selector(selector, color="yellow")
                    )
                return result

            async def get_children_tags_wrapper(selector: str, **kwargs):
                logger.debug(f"Agent calling get_children_tags: '{selector}'")
                result = await tools_instance.get_children_tags(selector=selector, **kwargs)
                # Highlight the parent element being inspected
                if result and result.parent_found and not result.error:
                    logger.debug(f"Highlighting parent for get_children_tags: '{selector}'")
                    asyncio.create_task(self._highlight_elements_by_selector(selector, color="red"))
                return result

            async def get_siblings_wrapper(selector: str, **kwargs):
                logger.debug(f"Agent calling get_siblings: '{selector}'")
                result = await tools_instance.get_siblings(selector=selector, **kwargs)
                # Highlight the element whose siblings are being checked
                if result and result.element_found and not result.error:
                    logger.debug(f"Highlighting element for get_siblings: '{selector}'")
                    asyncio.create_task(
                        self._highlight_elements_by_selector(selector, color="blue")
                    )
                return result

            async def extract_data_from_element_wrapper(selector: str, **kwargs):
                # This one we MIGHT want to highlight differently upon final success?
                # For now, just log.
                logger.debug(f"Agent calling extract_data_from_element: '{selector}'")
                return await tools_instance.extract_data_from_element(selector=selector, **kwargs)

            wrapped_tools = [
                Tool(evaluate_selector_wrapper),
                Tool(get_children_tags_wrapper),
                Tool(get_siblings_wrapper),
                Tool(extract_data_from_element_wrapper),
            ]

            # --- Setup System Prompt --- #
            system_prompt = _SYSTEM_PROMPT_BASE
            if current_dom_string:
                # Limit DOM representation size if needed (copied logic)
                max_dom_len = 10000
                truncated_dom = current_dom_string[:max_dom_len]
                if len(current_dom_string) > max_dom_len:
                    truncated_dom += "\n... (truncated)"
                system_prompt += _DOM_CONTEXT_PROMPT_SECTION.format(
                    dom_representation=truncated_dom
                )
            elif not self._active_tab_dom_string:  # Log only if it was never fetched
                logger.warning(f"Proceeding without DOM string representation for tab {tab_ref.id}")

            agent = Agent(
                "anthropic:claude-3-7-sonnet-latest",
                output_type=SelectorProposal,
                tools=wrapped_tools,
                system_prompt=system_prompt,
            )

            logger.info("Starting agent.run()...")
            # Construct the agent query (similar to SelectorAgent.find_and_extract)
            query_parts = [
                f"Generate the most STABLE CSS selector for the element described as '{target_description}'."
            ]
            # Add other parts based on extraction needs later if desired
            query_parts.append(
                "Prioritize stable attributes and classes, AVOIDING generated IDs/classes like 'emberXXX' or random strings."
            )
            query_parts.append("Follow the mandatory workflow strictly.")
            # --- Add explicit output format instruction ---
            query_parts.append(
                "CRITICAL: Your FINAL output MUST be a single JSON object conforming EXACTLY to the SelectorProposal schema. "
                "This JSON object MUST include values for the fields: 'proposed_selector' (string) and 'reasoning' (string). "
                "DO NOT include other fields like 'final_verification' or 'extraction_result' in the final JSON output."
            )
            # --- End explicit output instruction ---
            query = " ".join(query_parts)

            agent_run_result = await agent.run(query)

            # --- Process Result --- #
            if isinstance(agent_run_result.output, SelectorProposal):
                proposal = agent_run_result.output
                logger.info(
                    f"FINISHED. Proposal: {proposal.proposed_selector}, Reason: {proposal.reasoning}"
                )
                verification_result: Optional[SelectorEvaluationResult] = None
                try:
                    # Use the existing tools instance to evaluate
                    verification_result = await tools_instance.evaluate_selector(
                        selector=proposal.proposed_selector,
                        target_text_to_check="",  # No specific text needed for final check
                    )
                except Exception as verification_e:
                    logger.error(
                        f"Error during CLI verification of '{proposal.proposed_selector}': {verification_e}",
                        exc_info=True,
                    )

                # --- Conditional Highlight --- #
                if (
                    verification_result
                    and verification_result.element_count == 1
                    and not verification_result.error
                    and not verification_result.size_validation_error
                ):
                    logger.info(
                        f"CLI verification successful. Highlighting final unique selector: '{proposal.proposed_selector}' green..."
                    )
                    # Use fire and forget for the final highlight as well
                    asyncio.create_task(
                        self._highlight_elements_by_selector(
                            proposal.proposed_selector, color="lime"
                        )
                    )
                elif verification_result:
                    logger.warning(
                        f"CLI verification failed or selector not unique (count={verification_result.element_count}, error='{verification_result.error}', size_error='{verification_result.size_validation_error}'). Not highlighting final."
                    )
                    # --- Clear state if verification fails ---
                    self._highlights_active = False
                    self._last_highlight_selector = None
                    self._last_highlight_color = None
                    # --- End clear state ---
                else:
                    logger.error(
                        "CLI verification step failed unexpectedly. Not highlighting final."
                    )
                    # --- Clear state if verification fails ---
                    self._highlights_active = False
                    self._last_highlight_selector = None
                    self._last_highlight_color = None
                    # --- End clear state ---

            else:
                logger.error(
                    f"Agent returned unexpected output type: {type(agent_run_result.output)} / {agent_run_result.output}"
                )

        except Exception as e:
            logger.error(
                f"Error running SelectorAgent for target '{target_description}': {e}",
                exc_info=True,
            )

    async def _highlight_elements_by_selector(self, selector: str, color: str = "yellow") -> None:
        """Highlights elements matching a selector with a specific color using overlays."""
        logger.debug(f"Request to highlight: '{selector}' with color {color}")

        # --- Determine alternating color logic --- #
        current_color = color
        alternate_color_map = {
            "yellow": "orange",
            "blue": "purple",  # Added blue -> purple
            "red": "brown",  # Added red -> brown
            # Add other base colors and their alternates if needed
        }
        # Check if the requested color is a base color that should alternate
        # and if the last highlight used that *same* base color.
        if color in alternate_color_map and self._last_highlight_color == color:
            current_color = alternate_color_map[color]
            logger.debug(f"Alternating highlight from {color} to {current_color}.")
        # --- End alternate color logic --- #

        # --- Store state for re-highlighting ---
        self._last_highlight_selector = selector
        self._last_highlight_color = current_color  # Store the actual color used
        self._highlights_active = True
        # --- End store state ---

        # --- Clear previous highlights FIRST ---
        await self._clear_all_highlights()  # Ensure this is called
        # --- End clear previous ---

        if not self._active_tab_ref or not self._active_tab_ref.ws_url:
            logger.warning("Cannot highlight selector: No active tab reference or websocket URL.")
            return

        ws_url = self._active_tab_ref.ws_url
        url = self._active_tab_ref.url
        tab_id = self._active_tab_ref.id
        logger.info(
            f"Attempting to highlight selector '{selector}' on tab {tab_id} with color {color}"
        )

        # Escape the selector string for use within the JS string literal
        escaped_selector = (
            selector.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("`", "\\`")
        )

        highlight_style = f"2px solid {current_color}"  # Use current_color
        background_color = current_color + "33"  # Use current_color
        container_id = "selectron-highlight-container"
        overlay_attribute = "data-selectron-highlight-overlay"

        js_code = f"""
        (function() {{
            const selector = `{escaped_selector}`;
            const borderStyle = '{highlight_style}';
            const bgColor = '{background_color}';
            const containerId = '{container_id}';
            const overlayAttr = '{overlay_attribute}';

            // Find or create the container
            let container = document.getElementById(containerId);
            if (!container) {{
                container = document.createElement('div');
                container.id = containerId;
                container.style.position = 'fixed';
                container.style.pointerEvents = 'none';
                container.style.top = '0';
                container.style.left = '0';
                container.style.width = '100%';
                container.style.height = '100%';
                container.style.zIndex = '2147483647'; // Max z-index
                container.style.backgroundColor = 'transparent';
                document.body.appendChild(container);
            }}

            const elements = document.querySelectorAll(selector);
            if (!elements || elements.length === 0) {{
                return `No elements found for selector: ${{selector}}`;
            }}

            let highlightedCount = 0;
            elements.forEach(el => {{
                try {{
                    const rects = el.getClientRects();
                    if (!rects || rects.length === 0) return; // Skip elements without geometry

                    for (const rect of rects) {{
                        if (rect.width === 0 || rect.height === 0) continue; // Skip empty rects

                        const overlay = document.createElement('div');
                        overlay.setAttribute(overlayAttr, 'true'); // Mark as overlay
                        overlay.style.position = 'fixed';
                        overlay.style.border = borderStyle;
                        overlay.style.backgroundColor = bgColor;
                        overlay.style.pointerEvents = 'none';
                        overlay.style.boxSizing = 'border-box';
                        overlay.style.top = `${{rect.top}}px`;
                        overlay.style.left = `${{rect.left}}px`;
                        overlay.style.width = `${{rect.width}}px`;
                        overlay.style.height = `${{rect.height}}px`;
                        overlay.style.zIndex = '2147483647'; // Ensure overlay is on top

                        container.appendChild(overlay);
                    }}
                    highlightedCount++;
                }} catch (e) {{
                     console.warn('Selectron highlight error for one element:', e);
                }}
            }});

            return `Highlighted ${{highlightedCount}} element(s) (using overlays) for: ${{selector}}`;
        }})();
        """

        try:
            executor = CdpBrowserExecutor(ws_url, url)
            result = await executor.evaluate(js_code)
            logger.info(f"Highlight JS execution result: {result}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"Highlight selector failed for tab {tab_id}: WebSocket error - {e}")
        except Exception as e:
            logger.error(
                f"Highlight selector failed for tab {tab_id}: Unexpected error - {e}",
                exc_info=True,
            )

    async def _clear_all_highlights(self) -> None:
        """Removes all highlights previously added by this tool."""
        if not self._active_tab_ref or not self._active_tab_ref.ws_url:
            # Don't warn here, this might be called when no tab is active
            return

        ws_url = self._active_tab_ref.ws_url
        url = self._active_tab_ref.url
        tab_id = self._active_tab_ref.id
        # logger.debug(f"Attempting to clear highlights on tab {tab_id}") # Debug log

        container_id = "selectron-highlight-container"

        js_code = f"""
        (function() {{
            const containerId = '{container_id}'; // Capture ID for message
            const container = document.getElementById(containerId);
            let count = 0;
            if (container) {{
                count = container.childElementCount; // Count overlays before removing
                try {{
                    container.remove(); // Remove the whole container
                    return `SUCCESS: Removed highlight container ('${{containerId}}') with ${{count}} overlays.`;
                }} catch (e) {{
                    return `ERROR: Failed to remove container ('${{containerId}}'): ${{e.message}}`;
                }}
            }} else {{
                return 'INFO: Highlight container not found, nothing to remove.';
            }}
        }})();
        """
        try:
            # Use CdpBrowserExecutor to handle connection and execution
            # logger.info(f"Attempting to clear highlights on tab {tab_id}...") # Changed to info
            executor = CdpBrowserExecutor(ws_url, url)
            _result = await executor.evaluate(js_code)
            # logger.info(f"Clear highlights JS result: {result}") # Changed to info
        except websockets.exceptions.WebSocketException:
            # Ignore connection errors during cleanup, tab might be closed
            pass
        except Exception as e:
            logger.warning(f"Non-critical error clearing highlights on tab {tab_id}: {e}")

    async def trigger_rehighlight(self):
        """Triggers a re-highlight using the last known selector and color."""
        # logger.debug("Triggering rehighlight") # Optional debug log
        if self._highlights_active and self._last_highlight_selector and self._last_highlight_color:
            # logger.debug(f"Rehighlighting '{self._last_highlight_selector}' with {self._last_highlight_color}")
            await self._highlight_elements_by_selector(
                self._last_highlight_selector, self._last_highlight_color
            )
        # else:
        # logger.debug("Skipping rehighlight (not active or no selector/color)")


if __name__ == "__main__":
    # Ensure logger is initialized (and file created) before app runs
    get_logger("__main__")  # Initial call to setup file logging
    app = CliApp()
    app.run()

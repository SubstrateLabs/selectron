import asyncio
import base64
import io
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import openai
import websockets
from markdownify import markdownify as md_converter
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image
from pydantic_ai import Agent, Tool
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    Markdown,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker

from selectron.ai.selector_agent import (
    DOM_CONTEXT_PROMPT,
    SYSTEM_PROMPT_BASE,
)
from selectron.ai.selector_tools import (
    SelectorEvaluationResult,
    SelectorTools,
)  # Added SelectorEvaluationResult back
from selectron.ai.selector_types import (
    SelectorProposal,
)
from selectron.chrome.cdp_executor import CdpBrowserExecutor  # Added back
from selectron.chrome.chrome_cdp import (
    ChromeTab,  # Added back
    capture_tab_screenshot,  # Added back
    get_final_url_and_title,  # Added back
    get_html_via_ws,  # Added back
    wait_for_page_load,  # Added back
)
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent  # Added back
from selectron.chrome.connect import ensure_chrome_connection  # Added back
from selectron.chrome.highlight_service import HighlightService  # Added back
from selectron.chrome.types import TabReference  # Added back
from selectron.dom.dom_attributes import DOM_STRING_INCLUDE_ATTRIBUTES  # Added back
from selectron.dom.dom_service import DomService  # Added back
from selectron.util.extract_metadata import HtmlMetadata, extract_metadata  # Added back
from selectron.util.get_app_dir import get_app_dir  # Added back
from selectron.util.logger import get_logger
from selectron.util.sample_save import save_sample_data  # Added back

# --- End Selectron Imports ---

# --- Logging Setup ---
logger = get_logger(__name__)
# Set library levels (consider moving this to get_logger if applicable globally)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)  # Adjusted level

# Define log file path using get_app_dir (ensure consistency with logger.py)
LOG_FILE = get_app_dir() / "selectron.log"

# --- Textual App ---


class Selectron(App[None]):
    CSS_PATH = "cli.tcss"  # Add path to the CSS file
    BINDINGS = [
        Binding(key="ctrl+c", action="quit", description="Quit App", show=False),
        Binding(key="ctrl+q", action="quit", description="Quit App", show=True),
        Binding(key="ctrl+l", action="open_log_file", description="Open Logs", show=True),
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
    highlight_service: HighlightService
    _openai_client: Optional[openai.AsyncOpenAI] = None
    _proposal_tasks: dict[str, asyncio.Task] = {}
    _last_proposal_status: dict[str, str] = {}
    _last_proposed_selector: Optional[str] = None

    def __init__(self):
        super().__init__()
        self.shutdown_event = asyncio.Event()
        # Initialize log watching attributes
        self._log_file_path = LOG_FILE
        self._last_log_position = 0
        # Ensure log file exists (logger.py should also do this)
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file_path.touch(exist_ok=True)  # Explicitly create if doesn't exist
        self.highlight_service = HighlightService()

    @property
    def log_panel(self) -> RichLog:
        """Convenience property to access the RichLog widget."""
        return self.query_one("#log-panel", RichLog)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        # Main container holds the TabbedContent
        with Container(id="main-container"):  # Give the container an ID for potential styling
            # TabbedContent takes up the main area
            with TabbedContent(initial="logs-tab"):
                # Logs Tab
                with TabPane("✧ Logs ✧", id="logs-tab"):
                    yield RichLog(highlight=True, markup=False, id="log-panel", wrap=True)
                # List View Tab
                with TabPane("✧ Data ✧", id="list-tab"):
                    yield ListView(id="placeholder-list")
        # Input bar remains docked at the bottom, outside the main container
        with Horizontal(classes="input-bar"):  # Container for input and submit
            yield Input(placeholder="Enter description or let AI propose...", id="log-input")
            yield Button("Select", id="submit-button")

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

        # Initialize OpenAI client
        try:
            self._openai_client = openai.AsyncOpenAI()
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            # Decide if we should exit or just log the error
            # self.exit("Failed to initialize OpenAI client.") # Optional: exit if client is crucial

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

    @on(Input.Submitted, "#log-input")  # Handle Enter key on the specific input
    def handle_log_input_submission(self, event: Input.Submitted) -> None:
        """Logs the input field's content when Enter is pressed."""
        if event.value:  # Only log if there's text
            description = event.value
            # Use the refactored helper method to trigger the agent
            self.call_later(self._trigger_agent_run, description)
            event.input.clear()  # Clear the input after triggering
        else:
            logger.debug(
                "Log input submitted but was empty."
            )  # Optional: log empty submissions as debug

    @on(Button.Pressed, "#submit-button")
    async def handle_submit_button_press(self, event: Button.Pressed) -> None:
        """Handles the submit button press, triggering the agent with the input content."""
        input_widget = self.query_one("#log-input", Input)
        description = input_widget.value
        if description:
            # Use the refactored helper method to trigger the agent
            await self._trigger_agent_run(description)
            input_widget.clear()  # Clear the input after triggering
        else:
            logger.debug("Submit button pressed with empty input.")

    async def _trigger_agent_run(self, description: str):
        """Triggers the agent execution with the given description."""
        if not description:
            logger.debug("Agent trigger attempted with empty description.")
            return

        logger.info(f"Triggering Agent Run with: '{description}'")

        # --- Cancel previous agent, clear state, and clear highlights ---
        if self._agent_worker and self._agent_worker.is_running:
            logger.debug("Cancelling previous agent worker.")
            self._agent_worker.cancel()
        # Clear highlight state via service
        self.highlight_service.set_active(False)
        # Use call_later to ensure visual highlight clearing happens in the event loop
        self.call_later(self.highlight_service.clear, self._active_tab_ref)
        # --- Start new agent worker ---
        logger.debug(f"Starting new agent worker for '{description}'.")
        self._agent_worker = self.run_worker(
            self._run_agent_and_highlight(description), exclusive=False
        )  # Assign the new worker
        # Set highlight service inactive explicitly when starting agent
        self.highlight_service.set_active(False)

    async def run_monitoring(self) -> None:
        """Runs the main Chrome monitoring loop."""
        if not await ensure_chrome_connection():
            logger.error("Failed to establish Chrome connection.")
            return
        self.monitor = ChromeMonitor(
            rehighlight_callback=self.trigger_rehighlight,
            check_interval=1.5,
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
        # Immediately clear highlights if the active tab is closed or navigates away
        active_tab_closed_or_navigated = False
        if self._active_tab_ref:
            for closed_ref in event.closed_tabs:
                if closed_ref.id == self._active_tab_ref.id:
                    active_tab_closed_or_navigated = True
                    break
            if not active_tab_closed_or_navigated:
                for _navigated_tab, old_ref in event.navigated_tabs:
                    if old_ref.id == self._active_tab_ref.id:
                        active_tab_closed_or_navigated = True
                        break

        # --- Cancel Stale Proposal Tasks ---
        tabs_navigated_away_from_or_closed = list(event.closed_tabs) + [
            old_ref for _, old_ref in event.navigated_tabs
        ]
        for ref in tabs_navigated_away_from_or_closed:
            if ref.id in self._proposal_tasks:
                task_to_cancel = self._proposal_tasks.pop(ref.id, None)  # Use pop with default None
                if task_to_cancel and not task_to_cancel.done():
                    logger.info(
                        f"Polling detected navigation/closure for tab {ref.id}. Cancelling stale proposal task."
                    )
                    task_to_cancel.cancel()
                elif task_to_cancel:  # Task existed but was already done
                    logger.debug(
                        f"Proposal task for navigated/closed tab {ref.id} was already finished."
                    )

        if active_tab_closed_or_navigated:
            logger.info("Active tab closed or navigated, clearing highlights.")
            await self.highlight_service.clear(self._active_tab_ref)
            self.highlight_service.set_active(False)
            # self._active_tab_ref = None # Consider clearing ref here?
            # self._active_tab_dom_string = None

        for new_tab in event.new_tabs:
            if new_tab.webSocketDebuggerUrl:
                logger.info(f"Polling: New Tab: {new_tab.title} ({new_tab.url})")
                tasks.append(
                    self._process_new_tab(new_tab)
                )  # This will now create proposal tasks internally
            else:
                logger.warning(f"Polling: New tab {new_tab.id} missing websocket URL.")

        for closed_ref in event.closed_tabs:
            logger.info(f"Polling: Closed Tab: ID {closed_ref.id} ({closed_ref.url})")

        for navigated_tab, old_ref in event.navigated_tabs:
            logger.info(
                f"Polling: Navigated Tab: ID {navigated_tab.id} from {old_ref.url} TO {navigated_tab.url}"
            )
            if navigated_tab.webSocketDebuggerUrl:
                tasks.append(
                    self._process_new_tab(navigated_tab)
                )  # This will now create proposal tasks internally
            else:
                logger.warning(f"Polling: Navigated tab {navigated_tab.id} missing websocket URL.")

        if tasks:
            await asyncio.gather(*tasks)  # Run processing for new/navigated tabs

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
        logger.info(f"Interaction: Content Fetched: Tab {ref.id} ({ref.url}) ScrollY: {scroll_y}")
        self._active_tab_ref = ref  # Keep updating active tab ref

        # --- Add Debugging Log ---
        logger.debug(
            f"Checking retry condition for Tab {ref.id}: "
            f"Last Status = '{self._last_proposal_status.get(ref.id)}', "
            f"Task Running = {ref.id in self._proposal_tasks}"
        )
        # --- End Debugging Log ---

        # --- Attempt Proposal Retry if applicable ---
        # Check if last status was 'not_found' and no proposal task is currently running for this tab
        if (
            self._last_proposal_status.get(ref.id) == "not_found"
            and ref.id not in self._proposal_tasks
        ):
            logger.info(
                f"Retrying proposal generation for tab {ref.id} after interaction (previous status was 'not_found')."
            )
            # Check prerequisites again
            if self._openai_client and image:
                # Create and store the new proposal task (using the same logic as in _process_new_tab)
                logger.info(
                    f"Creating proposal generation task for tab {ref.id} ({ref.url}) (Retry)"
                )
                proposal_task = asyncio.create_task(
                    self._run_proposal_generation(
                        tab_id=ref.id,
                        final_url=ref.url,
                        screenshot_pil_image=image,  # Use the new image from interaction
                    )
                )
                self._proposal_tasks[ref.id] = proposal_task

                # Re-use the same callback structure
                def _proposal_done_callback(task: asyncio.Task, tab_id_cb: str):
                    try:
                        exc = task.exception()
                        if exc:
                            if isinstance(exc, asyncio.CancelledError):
                                logger.debug(
                                    f"Proposal task (Retry) for tab {tab_id_cb} was cancelled."
                                )
                            else:
                                logger.error(
                                    f"Proposal task (Retry) for tab {tab_id_cb} failed: {exc}",
                                    exc_info=exc,
                                )
                    finally:
                        removed_task = self._proposal_tasks.pop(tab_id_cb, None)
                        if removed_task:
                            logger.debug(
                                f"Removed completed/cancelled proposal task (Retry) for tab {tab_id_cb}."
                            )

                proposal_task.add_done_callback(lambda t: _proposal_done_callback(t, ref.id))
            else:
                # Log if retry prerequisites are missing
                missing_prereqs = []
                if not self._openai_client:
                    missing_prereqs.append("OpenAI client")
                if not image:
                    missing_prereqs.append("Screenshot")
                logger.warning(
                    f"Skipping proposal retry for tab {ref.id} due to missing prerequisites: {', '.join(missing_prereqs)}"
                )
        # --- End Proposal Retry ---

        # Log if DOM string was fetched
        if dom_string:
            logger.info(f"    Fetched DOM (Length: {len(dom_string)})")
            self._active_tab_dom_string = dom_string  # Update DOM string based on interaction fetch
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

        # --- Update Active Tab Reference (Keep this logic) --- #
        # Set the active tab reference ONLY if we successfully got the core data needed for the agent
        if final_url and html and ws_url:
            new_tab_ref = TabReference(
                id=tab.id, url=final_url, title=final_title, html=html, ws_url=ws_url
            )
            logger.info(
                f"Processing complete for tab {tab.id}, preparing to update active reference."
            )

            # --- Clear highlights if switching FROM a different active tab ---
            if self._active_tab_ref and self._active_tab_ref.id != new_tab_ref.id:
                logger.info(
                    f"Switching active tab from {self._active_tab_ref.id} to {new_tab_ref.id}. Clearing old highlights."
                )
                await self.highlight_service.clear(self._active_tab_ref)
            # --- End clear highlights ---

            # Now, update the active tab reference
            self._active_tab_ref = new_tab_ref
            self._active_tab_dom_string = dom_string  # Store the DOM string we fetched
            logger.info(f"Active tab reference UPDATED to: {tab.id} ({final_url})")

            # Ensure highlight state is inactive for the new context
            self.highlight_service.set_active(False)

        elif self._active_tab_ref and self._active_tab_ref.id == tab.id:
            # If this tab *was* the active one but processing failed, clear it
            logger.warning(f"Clearing active tab reference for {tab.id} due to processing failure.")
            self._active_tab_ref = None
            self._active_tab_dom_string = None

        # --- Generate Auto-Proposal (Task Based) --- #
        if self._openai_client and screenshot_pil_image:
            # --- Cancel previous task for the SAME tab ID if it exists (e.g., rapid navigation) ---
            if tab.id in self._proposal_tasks:
                existing_task = self._proposal_tasks.pop(tab.id)
                if not existing_task.done():
                    logger.warning(
                        f"Cancelling existing proposal task for tab {tab.id} due to new processing request."
                    )
                    existing_task.cancel()
            # --- End cancellation of previous ---

            logger.info(f"Creating proposal generation task for tab {tab.id} ({final_url})")
            proposal_task = asyncio.create_task(
                self._run_proposal_generation(
                    tab_id=tab.id, final_url=final_url, screenshot_pil_image=screenshot_pil_image
                )
            )
            self._proposal_tasks[tab.id] = proposal_task

            # Define the callback slightly differently to handle potential exceptions in pop
            def _proposal_done_callback(task: asyncio.Task, tab_id_cb: str):
                try:
                    # Log if the task raised an exception
                    exc = task.exception()
                    if exc:
                        if isinstance(exc, asyncio.CancelledError):
                            # Log cancellation at DEBUG level
                            logger.debug(
                                f"Proposal task for tab {tab_id_cb} was cancelled (expected)."
                            )
                        else:
                            # Log other exceptions at ERROR level
                            logger.error(
                                f"Proposal task for tab {tab_id_cb} failed with exception: {exc}",
                                exc_info=exc,
                            )
                finally:
                    # Always try to remove the task from the dict
                    removed_task = self._proposal_tasks.pop(tab_id_cb, None)
                    if removed_task:
                        logger.debug(
                            f"Removed completed/cancelled proposal task for tab {tab_id_cb} from tracking dict."
                        )

            # Add the callback to remove the task from the dictionary upon completion/cancellation
            proposal_task.add_done_callback(lambda t: _proposal_done_callback(t, tab.id))

        else:
            # Keep the logging for missing prerequisites
            missing_for_proposal = []
            if not self._openai_client:
                missing_for_proposal.append("OpenAI client")
            if not screenshot_pil_image:
                missing_for_proposal.append("Screenshot")
            if missing_for_proposal:
                logger.warning(
                    f"Skipping proposal generation task creation for {final_url or tab.url} due to missing: {', '.join(missing_for_proposal)}"
                )
        # --- End Generate Auto-Proposal --- #

        # --- Save Data (Keep this logic) ---
        if html and metadata and final_url:  # Keep checking for html/metadata for saving
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

        # --- Cancel any outstanding proposal tasks ---
        if self._proposal_tasks:
            logger.info(f"Cancelling {len(self._proposal_tasks)} outstanding proposal tasks...")
            tasks_to_cancel = list(self._proposal_tasks.values())  # Get tasks before clearing dict
            self._proposal_tasks.clear()  # Clear the dict
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            # Give cancellations a moment to propagate
            await asyncio.sleep(0.1)
        # --- End cancel proposal tasks ---

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
        # Clear highlight state via service
        self.highlight_service.set_active(False)
        # Clear highlights before exiting (fire and forget)
        self.call_later(self.highlight_service.clear, self._active_tab_ref)
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
        """Runs the SelectorAgent, captures screenshots on highlight, and highlights the final proposed selector."""
        if not self._active_tab_ref or not self._active_tab_ref.ws_url:
            logger.warning("Cannot run agent: No active tab reference with ws_url.")
            return

        latest_screenshot: Optional[Image.Image] = None  # Variable to hold the latest screenshot
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

            # --- Helper to Update List View --- #
            async def _update_list_with_markdown(html_snippets: list[str], source_selector: str):
                # Takes raw HTML snippets now
                try:
                    list_view = self.query_one("#placeholder-list", ListView)
                    await list_view.clear()
                    if not md_converter:
                        await list_view.append(
                            ListItem(Static("[markdownify library not installed]"))
                        )
                        return

                    if html_snippets:
                        logger.debug(
                            f"Updating list view with {len(html_snippets)} HTML snippets for '{source_selector}', converting to markdown."
                        )
                        for html_content in html_snippets:
                            try:
                                md_content = md_converter(html_content, heading_style="ATX")
                            except Exception as md_err:
                                logger.warning(
                                    f"Failed to convert HTML snippet to markdown: {md_err}"
                                )
                                md_content = f"_Error converting HTML:_\\n```html\\n{html_content[:200]}...\\n```"

                            md_widget = Markdown(md_content.strip(), classes="markdown-snippet")
                            list_item = ListItem(md_widget, classes="markdown-list-item")
                            await list_view.append(list_item)
                    else:
                        logger.debug(f"No HTML snippets to display for '{source_selector}'.")
                        await list_view.append(ListItem(Static("[No matching elements found]")))
                except Exception as list_update_err:
                    logger.error(f"Error updating list view: {list_update_err}")

            # --- End Helper --- #

            async def evaluate_selector_wrapper(selector: str, target_text_to_check: str, **kwargs):
                nonlocal latest_screenshot  # Allow modification
                # Run the actual tool
                logger.debug(f"Agent calling evaluate_selector: '{selector}'")
                result = await tools_instance.evaluate_selector(
                    selector=selector,
                    target_text_to_check=target_text_to_check,
                    **kwargs,
                    return_matched_html=True,  # Ensure we always request HTML
                )

                # Update list view with markdown from successful evaluations
                if result and not result.error:
                    # Pass raw HTML snippets to the helper for conversion
                    html_to_show = result.matched_html_snippets or []
                    await _update_list_with_markdown(html_to_show, selector)
                # Do not clear list view on evaluation error, keep previous state

                if result and result.element_count > 0 and not result.error:
                    logger.debug(f"Highlighting intermediate selector: '{selector}'")
                    # Await highlight and capture result
                    success, img = await self.highlight_service.highlight(
                        self._active_tab_ref, selector, color="yellow"
                    )
                    if success and img:
                        latest_screenshot = img  # Update latest screenshot
                return result

            async def get_children_tags_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot  # Allow modification
                logger.debug(f"Agent calling get_children_tags: '{selector}'")
                result = await tools_instance.get_children_tags(selector=selector, **kwargs)
                # Highlight the parent element being inspected
                if result and result.parent_found and not result.error:
                    logger.debug(f"Highlighting parent for get_children_tags: '{selector}'")
                    success, img = await self.highlight_service.highlight(
                        self._active_tab_ref, selector, color="red"
                    )
                    if success and img:
                        latest_screenshot = img  # Update latest screenshot
                return result

            async def get_siblings_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot  # Allow modification
                logger.debug(f"Agent calling get_siblings: '{selector}'")
                result = await tools_instance.get_siblings(selector=selector, **kwargs)
                # Highlight the element whose siblings are being checked
                if result and result.element_found and not result.error:
                    logger.debug(f"Highlighting element for get_siblings: '{selector}'")
                    success, img = await self.highlight_service.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                    if success and img:
                        latest_screenshot = img  # Update latest screenshot
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

            system_prompt = SYSTEM_PROMPT_BASE
            if current_dom_string:
                system_prompt += DOM_CONTEXT_PROMPT.format(dom_representation=current_dom_string)
            elif not self._active_tab_dom_string:  # Log only if it was never fetched
                logger.warning(f"Proceeding without DOM string representation for tab {tab_ref.id}")

            agent = Agent(
                "openai:gpt-4.1",
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

            # --- Prepare Agent Input (Text + Optional Image) --- #
            # Initialize with text query as default
            agent_input: Any = query
            # --- End Prepare Agent Input --- #

            # Use the prepared agent_input (which could be str or list)
            agent_run_result = await agent.run(agent_input)

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

                # --- Conditional Highlight & State Management --- #
                highlight_color = "lime"
                verification_passed = False

                if proposal.target_cardinality == "unique":
                    if (
                        verification_result
                        and verification_result.element_count == 1
                        and not verification_result.error
                        and not verification_result.size_validation_error
                    ):
                        logger.info(
                            f"CLI verification successful for UNIQUE target. Highlighting '{proposal.proposed_selector}' green..."
                        )
                        verification_passed = True
                    elif verification_result:
                        logger.warning(
                            f"CLI verification failed for UNIQUE target (count={verification_result.element_count}, error='{verification_result.error}', size_error='{verification_result.size_validation_error}'). Not highlighting final."
                        )
                    else:
                        logger.error(
                            "CLI verification step failed unexpectedly for UNIQUE target. Not highlighting final."
                        )
                elif proposal.target_cardinality == "multiple":
                    if (
                        verification_result
                        and verification_result.element_count > 0  # Check for > 0 instead of == 1
                        and not verification_result.error
                        and not verification_result.size_validation_error  # Still check size error if applicable? Maybe not for multiple.
                    ):
                        logger.info(
                            f"CLI verification successful for MULTIPLE targets (count={verification_result.element_count}). Highlighting '{proposal.proposed_selector}' green..."
                        )
                        verification_passed = True
                    elif verification_result:
                        logger.warning(
                            f"CLI verification failed for MULTIPLE targets (count={verification_result.element_count}, error='{verification_result.error}', size_error='{verification_result.size_validation_error}'). Not highlighting final."
                        )
                    else:
                        logger.error(
                            "CLI verification step failed unexpectedly for MULTIPLE target. Not highlighting final."
                        )
                else:
                    # Should not happen if pydantic validation works
                    logger.error(f"Unknown target_cardinality: {proposal.target_cardinality}")

                # --- Perform Highlight / Clear State --- #
                if verification_passed:
                    # Store the successful selector
                    self._last_proposed_selector = proposal.proposed_selector
                    logger.info(f"Storing successful selector: {self._last_proposed_selector}")

                    # Use highlight_service
                    logger.debug("Attempting final highlight and screenshot...")
                    # Await final highlight and potentially capture final screenshot
                    final_success, final_img = await self.highlight_service.highlight(
                        self._active_tab_ref, proposal.proposed_selector, color=highlight_color
                    )
                    if final_success:
                        logger.info(
                            f"Final highlight successful for selector: {proposal.proposed_selector}"
                        )
                        if final_img:
                            logger.info("Final screenshot captured.")
                            # Optional: Save or use final_img
                        else:
                            logger.warning("Final highlight succeeded but screenshot failed.")

                        # --- Log final selected content --- #
                        try:
                            logger.debug(
                                f"Extracting final content for selector: {proposal.proposed_selector}"
                            )
                            extraction_result = await tools_instance.extract_data_from_element(
                                selector=proposal.proposed_selector
                            )
                            if extraction_result and extraction_result.extracted_markdown:
                                # Truncate for logging
                                log_content = extraction_result.extracted_markdown[:500]
                                if len(extraction_result.extracted_markdown) > 500:
                                    log_content += "... (truncated)"
                                logger.info(f"Final Selected Content:\\n{log_content}")
                            elif extraction_result and extraction_result.error:
                                logger.warning(
                                    f"Failed to extract final content: {extraction_result.error}"
                                )
                            else:
                                logger.warning(
                                    "Final content extraction returned no content or error."
                                )
                        except Exception as final_extract_err:
                            logger.error(
                                f"Error during final content extraction: {final_extract_err}",
                                exc_info=True,
                            )
                        # --- End log final content --- #

                    else:
                        logger.error(
                            f"Final highlight FAILED for selector: {proposal.proposed_selector}"
                        )
                        # If highlight fails, maybe don't update list? Or clear selector?
                        # For now, let's still assume the selector is valid and update list
                        # Removed call
                        # logger.debug("Scheduling list view update even though highlight failed...")
                        # self.call_later(self._evaluate_and_update_list_view) # Keep this call

                else:
                    # Clear highlights and state via service if verification fails
                    logger.info("Clearing highlights and state due to verification failure.")
                    # Clear the stored selector if verification failed
                    self._last_proposed_selector = None  # Keep this
                    # Clear the list view as well
                    self.call_later(self._clear_list_view)  # Keep this call
                    await self.highlight_service.clear(self._active_tab_ref)
                    self.highlight_service.set_active(False)

            else:
                logger.error(
                    f"Agent returned unexpected output type: {type(agent_run_result.output)} / {agent_run_result.output}"
                )
                # Clear selector and list view if agent fails
                self._last_proposed_selector = None
                self.call_later(self._clear_list_view)

        except Exception as e:
            logger.error(
                f"Error running SelectorAgent for target '{target_description}': {e}",
                exc_info=True,
            )
            # Clear selector and list view on general agent error
            self._last_proposed_selector = None
            self.call_later(self._clear_list_view)

    async def trigger_rehighlight(self):
        """Triggers a re-highlight using the HighlightService."""
        await self.highlight_service.rehighlight(self._active_tab_ref)

    async def _run_proposal_generation(
        self, tab_id: str, final_url: Optional[str], screenshot_pil_image: Image.Image
    ):
        """Handles the OpenAI call for proposal and triggers agent if successful."""
        if not self._openai_client:  # Re-check client just in case
            logger.error("OpenAI client not available for proposal generation.")
            return

        logger.info(f"Starting proposal generation task for tab {tab_id} ({final_url})")
        self._last_proposal_status[tab_id] = "pending"  # Set status to pending

        try:
            # 1. Encode image
            buffered = io.BytesIO()
            img_to_save = screenshot_pil_image
            if img_to_save.mode == "RGBA":
                img_to_save = img_to_save.convert("RGB")
            img_to_save.save(buffered, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 2. Define prompt
            proposal_prompt = """You are an expert UI analyst. Analyze the provided screenshot. Perform the following steps:

1.  **Identify the Main Content Area:** Locate the primary region displaying core content, ignoring global headers/footers/navigation/sidebars.
2.  **Analyze Main Content Structure:** Determine if this area primarily consists of:
    a.  **Recurring Units:** Multiple distinct, visually similar, repeating elements forming a list, grid, or feed (e.g., posts, videos, products, comments, list items).
    b.  **Single Block:** One dominant block of continuous text and/or images, characteristic of a single article, blog post, documentation page, or similar static content page.
    c.  **Other/Ambiguous:** A simple form, landing page elements, highly varied content without clear repetition, etc.

3.  **Determine Output based on Step 2:**
    *   **Case (a) - Recurring Units:** Identify the MOST representative type. Provide ONE concise, generic description suitable for selecting **ALL instances** (e.g., "All primary posts in the feed", "Each result item in the list"). Output JSON: `{"status": "found", "description": "Your description here"}`
    *   **Case (b) - Single Block (Article-like):** This is NOT a list of recurring primary items. Output JSON: `{"status": "not_found"}`
    *   **Case (c) - Other/Ambiguous:** Treat as not having clear, primary recurring items. Output JSON: `{"status": "not_found"}`

CRITICAL: Output ONLY the JSON object (`{"status": "found", "description": "..."}` OR `{"status": "not_found"}`). No other text, labels, formatting, or explanation.

Example "found" output:
`{"status": "found", "description": "All primary posts/updates in the main feed area"}`

Example "not_found" output (for single article, ambiguous page, form, etc.):
`{"status": "not_found"}`"""

            # 3. Construct messages
            messages: list[ChatCompletionUserMessageParam] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": proposal_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ]

            # 4. Make API call
            logger.info(f"Calling OpenAI vision model for structured proposal (Tab: {tab_id})...")
            completion = await self._openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            await asyncio.sleep(0)  # Check for cancellation

            # 5. Parse and conditionally trigger
            response_content = completion.choices[0].message.content
            current_status = "error"  # Default status if parsing fails
            if response_content:
                logger.debug(f"Raw proposal JSON response (Tab: {tab_id}): {response_content}")
                try:
                    proposal_data = json.loads(response_content)
                    status = proposal_data.get("status")
                    description = proposal_data.get("description")

                    if status == "found" and isinstance(description, str) and description.strip():
                        proposed_description = description.strip()
                        logger.info(
                            f"Proposal status: found (Tab: {tab_id}). Desc: '{proposed_description}'"
                        )
                        logger.info(
                            f"Updating input field for tab {tab_id} with: '{proposed_description}'"
                        )
                        try:
                            # Check if this tab is STILL the active tab before updating input
                            if self._active_tab_ref and self._active_tab_ref.id == tab_id:
                                log_input_widget = self.query_one("#log-input", Input)
                                # Update the input widget's value
                                log_input_widget.value = proposed_description
                                # Optionally move focus back to the input
                                # self.call_later(log_input_widget.focus)
                            else:
                                logger.info(
                                    f"Proposal for tab {tab_id} succeeded, but it's no longer the active tab. Input not updated."
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to get log input widget to update value for tab {tab_id}: {e}",
                                exc_info=True,
                            )
                        current_status = "found"  # Update status
                    elif status == "not_found":
                        logger.info(
                            f"Proposal status: not_found (Tab: {tab_id}). Input not updated."
                        )
                        current_status = "not_found"  # Update status
                    else:
                        logger.warning(
                            f"Proposal JSON (Tab: {tab_id}) unexpected structure. Status='{status}', Desc type='{type(description)}'. Input not updated."
                        )
                        current_status = "error"  # Keep status as error

                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"Failed to parse JSON response (Tab: {tab_id}): {json_err}\nResponse: {response_content}. Input not updated.",
                    )
                    current_status = "error"  # Set status to error
            else:
                logger.warning(
                    f"Proposal model returned empty response (Tab: {tab_id}). Input not updated."
                )
                current_status = "error"  # Set status to error

            self._last_proposal_status[tab_id] = current_status  # Store final status

        except asyncio.CancelledError:
            logger.debug(f"Proposal generation task for tab {tab_id} cancelled (expected).")
            # Don't change status if cancelled, allows retry based on previous status
            raise
        except Exception as proposal_err:
            logger.error(
                f"Error during proposal generation task for tab {tab_id}: {proposal_err}",
                exc_info=True,
            )
            self._last_proposal_status[tab_id] = "error"  # Store error status
        # finally: # Removed finally block for status setting, doing it after processing now
        #    logger.debug(f"Proposal generation task for tab {tab_id} finished.")

    async def _clear_list_view(self) -> None:
        """Helper method to safely clear the list view."""
        try:
            list_view = self.query_one("#placeholder-list", ListView)
            await list_view.clear()
            logger.debug("List view cleared.")
        except Exception as e:
            logger.error(f"Failed to query or clear list view: {e}")


if __name__ == "__main__":
    # Ensure logger is initialized (and file created) before app runs
    get_logger("__main__")  # Initial call to setup file logging
    app = Selectron()
    app.run()

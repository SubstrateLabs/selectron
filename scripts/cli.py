import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import openai
import websockets
from markdownify import markdownify
from PIL import Image
from pydantic_ai import Agent, Tool
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    DataTable,
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

from selectron.ai.propose_select import propose_select
from selectron.ai.selector_agent import (
    DOM_CONTEXT_PROMPT,
    SYSTEM_PROMPT_BASE,
)
from selectron.ai.selector_tools import (
    SelectorTools,
)
from selectron.ai.types import (
    ProposalResult,
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
from selectron.chrome.chrome_highlighter import ChromeHighlighter
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent
from selectron.chrome.connect import ensure_chrome_connection
from selectron.chrome.types import TabReference
from selectron.dom.dom_attributes import DOM_STRING_INCLUDE_ATTRIBUTES
from selectron.dom.dom_service import DomService
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger

logger = get_logger(__name__)
LOG_FILE = get_app_dir() / "selectron.log"


class Selectron(App[None]):
    CSS_PATH = "cli.tcss"
    BINDINGS = [
        Binding(key="ctrl+c", action="quit", description="Quit App", show=False),
        Binding(key="ctrl+q", action="quit", description="Quit App", show=True),
        Binding(key="ctrl+l", action="open_log_file", description="Open Logs", show=True),
    ]

    monitor: Optional[ChromeMonitor] = None
    monitor_task: Optional[Worker[None]] = None
    shutdown_event: asyncio.Event

    # Attributes for log file watching
    _log_file_path: Path
    _last_log_position: int
    _active_tab_ref: Optional[TabReference] = None
    _active_tab_dom_string: Optional[str] = None
    _agent_worker: Optional[Worker[None]] = None
    _highlighter: ChromeHighlighter
    _openai_client: Optional[openai.AsyncOpenAI] = None
    _proposal_tasks: dict[str, asyncio.Task] = {}
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
        self._highlighter = ChromeHighlighter()

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
                with TabPane("✦ Logs ✧", id="logs-tab"):
                    yield RichLog(highlight=True, markup=False, id="log-panel", wrap=True)
                with TabPane("✦ Extracted Markdown ✧", id="markdown-tab"):
                    yield ListView(id="markdown-list")
                with TabPane("✦ Parsed Data ✧", id="table-tab"):
                    yield DataTable(id="data-table")
        # Input bar remains docked at the bottom, outside the main container
        with Horizontal(classes="input-bar"):  # Container for input and submit
            yield Input(placeholder="Enter description or let AI propose...", id="prompt-input")
            yield Button("Select", id="submit-button")

        yield Footer()

    async def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Manually clear log file at the start of mounting to ensure clean slate
        try:
            # Opening with 'w' mode truncates the file.
            open(self._log_file_path, "w").close()  # Open in write mode and immediately close.
            self._last_log_position = 0  # Reset position after clearing
        except Exception as e:
            print(
                f"ERROR: Failed to clear log file {self._log_file_path} on mount: {e}",
                file=sys.stderr,
            )

        try:
            self._openai_client = openai.AsyncOpenAI()
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        self._load_initial_logs()
        self.set_interval(0.5, self._watch_log_file)  # Check every 500ms

        # --- Setup DataTable --- #
        try:
            table = self.query_one(DataTable)
            table.cursor_type = "row"
            table.add_column("Raw HTML", key="html_content")
            logger.debug("DataTable initialized with 'Raw HTML' column.")
        except Exception as table_init_err:
            logger.error(f"Failed to initialize DataTable: {table_init_err}", exc_info=True)
        # --- End Setup DataTable --- #

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

    @on(Input.Submitted, "#prompt-input")
    def handle_prompt_submitted(self, event: Input.Submitted) -> None:
        if event.value:
            description = event.value
            self.call_later(self._trigger_agent_run, description)
            event.input.clear()

    @on(Button.Pressed, "#submit-button")
    async def handle_submit_button_pressed(self, event: Button.Pressed) -> None:
        input_widget = self.query_one("#prompt-input", Input)
        description = input_widget.value
        if description:
            await self._trigger_agent_run(description)
            input_widget.clear()

    async def _trigger_agent_run(self, description: str):
        if self._agent_worker and self._agent_worker.is_running:
            logger.debug("Cancelling previous agent worker.")
            self._agent_worker.cancel()
        self._highlighter.set_active(False)
        self.call_later(self._highlighter.clear, self._active_tab_ref)
        logger.debug(f"Starting new agent worker for '{description}'.")
        self._agent_worker = self.run_worker(
            self._run_agent_and_highlight(description), exclusive=False
        )
        self._highlighter.set_active(False)

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
            await self._highlighter.clear(self._active_tab_ref)
            self._highlighter.set_active(False)
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

        # Log if DOM string was fetched
        if dom_string:
            logger.info(f"    Fetched DOM (Length: {len(dom_string)}) ")
            self._active_tab_dom_string = dom_string  # Update DOM string based on interaction fetch
        else:
            logger.warning(f"    DOM string not fetched for {ref.url} after interaction.")

        if ref.html:
            try:
                # Only update if this is the *currently active* tab visually
                if self._active_tab_ref and self._active_tab_ref.id == ref.id:
                    page_markdown = markdownify(ref.html, heading_style="ATX")
                    list_view = self.query_one("#markdown-list", ListView)
                    # Decide whether to clear or prepend/replace
                    # For now, let's clear and set, assuming interaction fetch should refresh the base view
                    await list_view.clear()
                    md_widget = Markdown(page_markdown.strip(), classes="full-page-markdown")
                    list_item = ListItem(md_widget, classes="markdown-list-item full-page-item")
                    await list_view.append(list_item)
                    logger.debug(
                        f"Updated initial page markdown for tab {ref.id} via interaction fetch"
                    )
            except Exception as page_md_err:
                logger.error(
                    f"Failed to update initial page markdown for tab {ref.id} via interaction fetch: {page_md_err}"
                )

        if ref.html:
            try:
                # Only update if this is the *currently active* tab visually
                if self._active_tab_ref and self._active_tab_ref.id == ref.id:
                    table = self.query_one(DataTable)
                    table.clear()
                    table.add_row(ref.html, key=f"tab_{ref.id}_html")
                    logger.debug(
                        f"Updated HTML content in table for tab {ref.id} via interaction fetch"
                    )
            except Exception as table_update_err:
                logger.error(
                    f"Failed to update HTML in table for tab {ref.id} via interaction fetch: {table_update_err}"
                )

        if not ref.html:
            logger.warning(f"    HTML not fetched for {ref.url}, skipping metadata/save.")

    async def _process_new_tab(self, tab: ChromeTab):
        """Fetches HTML, metadata, screenshot for a NEW or NAVIGATED tab and saves samples. (Instance Method)"""
        html = screenshot_pil_image = ws = dom_string = None
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
                if html:
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

        if final_url:
            new_tab_ref = TabReference(
                id=tab.id, url=final_url, title=final_title, html=html, ws_url=ws_url
            )
            if self._active_tab_ref and self._active_tab_ref.id != new_tab_ref.id:
                await self._highlighter.clear(self._active_tab_ref)
            self._active_tab_ref = new_tab_ref
            self._active_tab_dom_string = dom_string  # Store the DOM string we fetched
            logger.info(f"Active tab UPDATED to: {tab.id} ({final_url})")
            if html:
                try:
                    page_markdown = markdownify(html, heading_style="ATX")
                    list_view = self.query_one("#markdown-list", ListView)
                    await list_view.clear()  # Clear previous content
                    md_widget = Markdown(page_markdown.strip(), classes="full-page-markdown")
                    list_item = ListItem(md_widget, classes="markdown-list-item full-page-item")
                    await list_view.append(list_item)
                    logger.debug(f"Set initial page markdown for tab {tab.id}")
                except Exception as page_md_err:
                    logger.error(
                        f"Failed to set initial page markdown for tab {tab.id}: {page_md_err}"
                    )
            self._highlighter.set_active(False)

        elif self._active_tab_ref and self._active_tab_ref.id == tab.id:
            # If this tab *was* the active one but processing failed, clear it
            logger.warning(f"Clearing active tab reference for {tab.id} due to processing failure.")
            self._active_tab_ref = None
            self._active_tab_dom_string = None

        if self._openai_client and screenshot_pil_image:
            # --- Cancel previous task for the SAME tab ID if it exists (e.g., rapid navigation) ---
            if tab.id in self._proposal_tasks:
                existing_task = self._proposal_tasks.pop(tab.id)
                if not existing_task.done():
                    logger.warning(
                        f"Cancelling existing proposal task for tab {tab.id} due to new processing request."
                    )
                    existing_task.cancel()
            proposal_task = asyncio.create_task(
                self._run_proposal_generation(
                    tab_id=tab.id, final_url=final_url, screenshot_pil_image=screenshot_pil_image
                )
            )
            self._proposal_tasks[tab.id] = proposal_task

            def _proposal_done_callback(task: asyncio.Task, tab_id_cb: str):
                try:
                    # Check for cancellation FIRST to avoid noisy logs for expected cancellations
                    if task.cancelled():
                        logger.debug(f"Proposal task for tab {tab_id_cb} was cancelled (expected).")
                        return  # Don't proceed further if cancelled
                    # Log if the task raised an exception (other than cancellation)
                    exc = task.exception()
                    if exc:
                        logger.error(
                            f"Proposal task for tab {tab_id_cb} failed with exception: {exc}",
                            exc_info=exc,
                        )
                finally:
                    removed_task = self._proposal_tasks.pop(tab_id_cb, None)
                    if removed_task:
                        logger.debug(
                            f"Removed completed/cancelled proposal task for tab {tab_id_cb} from tracking dict."
                        )

            # remove the task from the dictionary upon completion/cancellation
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
        self._highlighter.set_active(False)
        # Clear highlights before exiting (fire and forget)
        self.call_later(self._highlighter.clear, self._active_tab_ref)
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
                # Fetch DOM String using the same connection
                try:
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
                    list_view = self.query_one("#markdown-list", ListView)
                    await list_view.clear()

                    if html_snippets:
                        logger.debug(
                            f"Updating list view with {len(html_snippets)} HTML snippets for '{source_selector}', converting to markdown."
                        )
                        for html_content in html_snippets:
                            try:
                                md_content = markdownify(html_content, heading_style="ATX")
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
                    success, img = await self._highlighter.highlight(
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
                    success, img = await self._highlighter.highlight(
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
                    success, img = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                    if success and img:
                        latest_screenshot = img  # Update latest screenshot
                return result

            async def extract_data_from_element_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot  # Allow modification
                # This one we MIGHT want to highlight differently upon final success?
                # For now, just log.
                logger.debug(f"Agent calling extract_data_from_element: '{selector}'")
                # Highlight this selector as potentially the final one
                logger.debug(f"Highlighting potentially final selector: '{selector}'")
                success, img = await self._highlighter.highlight(
                    self._active_tab_ref,
                    selector,
                    color="lime",  # Use 'lime' for final/extraction step
                )
                if success and img:
                    latest_screenshot = img  # Update latest screenshot if highlight succeeds
                elif not success:
                    logger.warning(f"Highlight failed for extraction selector: '{selector}'")
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
                    f"FINISHED. Proposal: {proposal.proposed_selector}\nREASONING: {proposal.reasoning}"
                )
                self._last_proposed_selector = proposal.proposed_selector
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
        await self._highlighter.rehighlight(self._active_tab_ref)

    async def _run_proposal_generation(
        self, tab_id: str, final_url: Optional[str], screenshot_pil_image: Image.Image
    ):
        """Handles the OpenAI call for proposal using the extracted function and updates UI/state."""
        if not self._openai_client:  # Re-check client just in case
            logger.error("OpenAI client not available for proposal generation.")
            return

        logger.info(
            f"Starting proposal generation task for tab {tab_id} ({final_url}) via extracted function."
        )
        try:
            # Call the extracted function
            proposal_result: ProposalResult = await propose_select(
                client=self._openai_client,
                screenshot=screenshot_pil_image,
            )

            if proposal_result.description:
                proposed_description = proposal_result.description
                logger.info(f"Proposal found (Tab: {tab_id}). Desc: '{proposed_description}'")
                try:
                    # Check if this tab is STILL the active tab before updating input
                    if self._active_tab_ref and self._active_tab_ref.id == tab_id:
                        log_input_widget = self.query_one("#prompt-input", Input)
                        log_input_widget.value = proposed_description
                        # Optionally move focus back to the input
                    else:
                        logger.info(
                            f"Proposal for tab {tab_id} succeeded, but it's no longer the active tab. Input not updated."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to get log input widget to update value for tab {tab_id}: {e}",
                        exc_info=True,
                    )
            elif proposal_result.error_message:
                logger.error(
                    f"Proposal generation failed for tab {tab_id}. Error: {proposal_result.error_message}"
                )
            else:
                logger.error(
                    f"Proposal generation for tab {tab_id} returned unexpected empty result."
                )

        except asyncio.CancelledError:
            logger.debug(f"Proposal generation task for tab {tab_id} cancelled (expected).")
            # Don't change status if cancelled, allows retry based on previous status
            raise  # Re-raise cancellation
        except Exception as proposal_err:  # Catch potential errors in the call itself
            logger.error(
                f"Unexpected error during proposal task execution for tab {tab_id}: {proposal_err}",
                exc_info=True,
            )

    async def _clear_list_view(self) -> None:
        """Helper method to safely clear the list view."""
        try:
            list_view = self.query_one("#markdown-list", ListView)
            await list_view.clear()
            logger.debug("List view cleared.")
        except Exception as e:
            logger.error(f"Failed to query or clear list view: {e}")


if __name__ == "__main__":
    # Ensure logger is initialized (and file created) before app runs
    get_logger("__main__")  # Initial call to setup file logging
    app = Selectron()
    app.run()

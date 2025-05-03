import asyncio
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

from selectron.ai.selector_agent import (
    DOM_CONTEXT_PROMPT,
    SYSTEM_PROMPT_BASE,
)
from selectron.ai.selector_tools import (
    SelectorTools,
)
from selectron.ai.types import (
    SelectorProposal,
)
from selectron.chrome.cdp_executor import CdpBrowserExecutor
from selectron.chrome.chrome_cdp import (
    get_final_url_and_title,
    get_html_via_ws,
    wait_for_page_load,
)
from selectron.chrome.chrome_highlighter import ChromeHighlighter
from selectron.chrome.connect import ensure_chrome_connection
from selectron.chrome.types import TabReference
from selectron.dom.dom_attributes import DOM_STRING_INCLUDE_ATTRIBUTES
from selectron.dom.dom_service import DomService
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger
from selectron.util.open_log_file import open_log_file

logger = get_logger(__name__)
LOG_FILE = get_app_dir() / "selectron.log"


class Selectron(App[None]):
    CSS_PATH = "cli.tcss"
    BINDINGS = [
        Binding(key="ctrl+c", action="quit", description="Quit App", show=False),
        Binding(key="ctrl+q", action="quit", description="Quit App", show=True),
        Binding(key="ctrl+l", action="open_log_file", description="Open Logs", show=True),
    ]

    # Removed monitor and monitor_task, will be managed by ChromeTabManager
    shutdown_event: asyncio.Event

    # Attributes for log file watching
    _log_file_path: Path
    _last_log_position: int
    _active_tab_ref: Optional[TabReference] = None
    _active_tab_dom_string: Optional[str] = None
    _agent_worker: Optional[Worker[None]] = None
    _highlighter: ChromeHighlighter
    _openai_client: Optional[openai.AsyncOpenAI] = None
    # Removed _proposal_tasks
    _last_proposed_selector: Optional[str] = None

    # Add instance for the new manager
    _tab_manager: Optional["ChromeTabManager"] = (
        None  # Forward reference if manager imports Selectron
    )

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
        # Tab manager will be initialized in on_mount

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

        logger.info("App mounted. Initializing Chrome tab manager...")
        # Initialize and start the tab manager
        self._tab_manager = ChromeTabManager(
            openai_client=self._openai_client,
            on_active_tab_updated=self._handle_active_tab_update,  # New callback handler
            on_page_content_ready=self._handle_page_content_update,  # New callback handler
            on_proposal_ready=self._handle_proposal_update,  # New callback handler
        )
        # Run the manager's monitoring loop in the background
        self.run_worker(
            self._tab_manager.run_monitoring_task(), exclusive=True, group="chrome_manager"
        )

        # Check connection (can be moved inside manager start later)
        if not await ensure_chrome_connection():
            logger.error("Failed to establish Chrome connection.")
            # Consider exiting or notifying user

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

    # --- Callback Handlers for ChromeTabManager --- #
    async def _handle_active_tab_update(
        self, tab_ref: Optional[TabReference], dom_string: Optional[str]
    ):
        """Handles updates to the active tab reference and DOM string from the manager."""
        logger.debug(f"Callback: Active tab updated to {tab_ref.id if tab_ref else 'None'}")
        self._active_tab_ref = tab_ref
        self._active_tab_dom_string = dom_string
        # If the tab is cleared, ensure highlights are cleared too.
        # The highlighter state might need careful management between agent runs and tab changes.
        if not tab_ref:
            logger.info("Active tab cleared by manager, clearing highlights.")
            await self._highlighter.clear(None)  # Clear generic highlights
            self._highlighter.set_active(False)

    async def _handle_page_content_update(self, html_content: str):
        """Handles updates to the page content (HTML) from the manager."""
        logger.debug(
            f"Callback: Page content update received (length: {len(html_content)}). Updating UI."
        )
        # Update Markdown View
        try:
            page_markdown = markdownify(html_content, heading_style="ATX")
            list_view = self.query_one("#markdown-list", ListView)
            await list_view.clear()
            md_widget = Markdown(page_markdown.strip(), classes="full-page-markdown")
            list_item = ListItem(md_widget, classes="markdown-list-item full-page-item")
            await list_view.append(list_item)
        except Exception as md_err:
            logger.error(f"Failed to update markdown view from callback: {md_err}")

        # Update Table View
        try:
            table = self.query_one(DataTable)
            table.clear()
            # Consider adding URL/Title if available in tab_ref when active tab updates?
            table.add_row(html_content, key="current_tab_html")
        except Exception as table_err:
            logger.error(f"Failed to update data table from callback: {table_err}")

    async def _handle_proposal_update(self, description: str):
        """Handles the proposed description update from the manager."""
        logger.debug(f"Callback: Proposal received: '{description}'. Updating input.")
        # Update Input Widget only if the current tab is still active (manager doesn't know UI state)
        if self._active_tab_ref:
            try:
                input_widget = self.query_one("#prompt-input", Input)
                # Only update if the input is currently empty or hasn't been manually edited?
                # Or just always update?
                input_widget.value = description
                logger.info(
                    f"Updated prompt input with proposal for tab {self._active_tab_ref.id}."
                )
                # input_widget.focus() # Optional focus
            except Exception as input_err:
                logger.error(f"Failed to update input widget from callback: {input_err}")
        else:
            logger.info("Proposal received, but no active tab. Input not updated.")

    async def action_quit(self) -> None:
        """Action to quit the app."""
        logger.info("Shutdown requested...")
        self.shutdown_event.set()

        # Signal the tab manager to stop
        if self._tab_manager:
            logger.info("Stopping Chrome tab manager...")
            await self._tab_manager.stop_monitoring()
            logger.info("Chrome tab manager stopped.")

        # Wait for the manager worker to finish
        # This assumes stop_monitoring is effective and the worker group is set.
        # Textual handles worker cleanup on exit, but explicit waiting can be safer.
        workers = [w for w in self.workers if w.group == "chrome_manager"]
        if workers:
            logger.info("Waiting for manager worker...")
            try:
                await workers[0].wait()  # Wait for the first (should be only) manager worker
                logger.info("Manager worker finished.")
            except Exception as e:
                logger.error(f"Error waiting for manager worker: {e}")

        # --- Cancel agent worker, clear state, and clear highlights on exit --- #
        if self._agent_worker and self._agent_worker.is_running:
            logger.info("Cancelling agent worker on exit...")
            self._agent_worker.cancel()
        self._highlighter.set_active(False)
        # Use call_later for fire-and-forget clear, happens before exit
        if self._active_tab_ref:  # Only clear if there was an active tab
            self.call_later(self._highlighter.clear, self._active_tab_ref)
            await asyncio.sleep(0.1)  # Short delay for clear task

        logger.info("Exiting application.")
        self.app.exit()

    def action_open_log_file(self) -> None:
        open_log_file(self._log_file_path)

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

        # Now we should have HTML, proceed with agent logic
        logger.info(
            f"Running SelectorAgent logic for target '{target_description}' on tab {tab_ref.id}"
        )
        base_url_for_agent = current_url  # Use the potentially updated current_url
        if not base_url_for_agent:
            logger.error(f"Cannot run agent: Base URL is missing for tab {tab_ref.id}")
            return
        if not current_html:
            logger.error(
                f"Cannot run agent: HTML content is missing even after fetch attempt for tab {tab_ref.id}."
            )
            return

        try:
            tools_instance = SelectorTools(html_content=current_html, base_url=base_url_for_agent)
            async def _update_list_with_markdown(html_snippets: list[str], source_selector: str):
                try:
                    list_view = self.query_one("#markdown-list", ListView)
                    await list_view.clear()
                    if html_snippets:
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
                        await list_view.append(ListItem(Static("[No matching elements found]")))
                except Exception as list_update_err:
                    logger.error(f"Error updating list view: {list_update_err}")

            async def evaluate_selector_wrapper(selector: str, target_text_to_check: str, **kwargs):
                nonlocal latest_screenshot  # Allow modification
                logger.debug(f"Agent calling evaluate_selector: '{selector}'")
                result = await tools_instance.evaluate_selector(
                    selector=selector,
                    target_text_to_check=target_text_to_check,
                    **kwargs,
                    return_matched_html=True, 
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
                    success, img = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                    if success and img:
                        latest_screenshot = img  # Update latest screenshot
                return result

            async def extract_data_from_element_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot  # Allow modification
                logger.debug(f"Highlighting potentially final selector: '{selector}'")
                success, img = await self._highlighter.highlight(
                    self._active_tab_ref,
                    selector,
                    color="lime",  
                )
                if success and img:
                    latest_screenshot = img 
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
            query = " ".join(query_parts)
            agent_input: Any = query

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
    # Import manager here to avoid circular dependency if manager needs Selectron types
    from selectron.chrome.chrome_tab_manager import ChromeTabManager

    app = Selectron()
    app.run()

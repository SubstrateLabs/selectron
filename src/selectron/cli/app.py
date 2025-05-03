import asyncio
from typing import Any, Optional

import openai
from markdownify import markdownify
from PIL import Image
from pydantic_ai import Agent, Tool
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
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker

from selectron.ai.propose_select import propose_select
from selectron.ai.selector_prompt import (
    SELECTOR_PROMPT_BASE,
    SELECTOR_PROMPT_DOM_TEMPLATE,
)
from selectron.ai.selector_tools import (
    SelectorTools,
)
from selectron.ai.types import (
    AutoProposal,
    SelectorProposal,
)
from selectron.chrome import chrome_launcher
from selectron.chrome.chrome_highlighter import ChromeHighlighter
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent
from selectron.chrome.types import TabReference
from selectron.cli.home_panel import ChromeStatus, HomePanel
from selectron.cli.log_panel import LogPanel
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger

logger = get_logger(__name__)
LOG_PATH = get_app_dir() / "selectron.log"
THEME_DARK = "nord"
THEME_LIGHT = "solarized-light"
DEFAULT_THEME = THEME_DARK


class SelectronApp(App[None]):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        Binding(key="ctrl+c", action="quit", description="⣏ Quit App ⣹", show=False),
        Binding(key="ctrl+q", action="quit", description="⣏ Quit App ⣹", show=True),
        Binding(key="ctrl+l", action="open_log_file", description="⣏ Open Logs ⣹", show=True),
        Binding(key="ctrl+t", action="toggle_dark", description="⣏ Light/Dark Theme ⣹", show=True),
    ]
    shutdown_event: asyncio.Event
    _active_tab_ref: Optional[TabReference] = None
    _active_tab_dom_string: Optional[str] = None
    _agent_worker: Optional[Worker[None]] = None
    _vision_worker: Optional[Worker[None]] = None
    _highlighter: ChromeHighlighter
    _openai_client: Optional[openai.AsyncOpenAI] = None
    _last_proposed_selector: Optional[str] = None
    _chrome_monitor: Optional[ChromeMonitor] = None

    def __init__(self):
        super().__init__()
        self.shutdown_event = asyncio.Event()
        self._highlighter = ChromeHighlighter()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main-container"):
            with TabbedContent(initial="home-tab"):
                with TabPane("⣏ Home ⣹", id="home-tab"):
                    yield HomePanel(id="home-panel-widget")
                with TabPane("⣏ Logs ⣹", id="logs-tab"):
                    yield LogPanel(log_file_path=LOG_PATH, id="log-panel-widget")
                with TabPane("⣏ Selected Markdown ⣹", id="markdown-tab"):
                    yield ListView(id="markdown-list")
                with TabPane("⣏ Parsed Data ⣹", id="table-tab"):
                    yield DataTable(id="data-table")
        with Horizontal(classes="input-bar"):
            yield Input(placeholder="Enter description or let AI propose...", id="prompt-input")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self._openai_client = openai.AsyncOpenAI()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        try:
            table = self.query_one(DataTable)
            table.cursor_type = "row"
            table.add_column("Raw HTML", key="html_content")
        except Exception as table_init_err:
            logger.error(f"Failed to initialize DataTable: {table_init_err}", exc_info=True)
        self.theme = DEFAULT_THEME

        self._chrome_monitor = ChromeMonitor(
            rehighlight_callback=self._handle_rehighlight,
            check_interval=2.0,
            interaction_debounce=0.7,
        )

        await self.action_check_chrome_status()

    async def _handle_polling_change(self, event: TabChangeEvent) -> None:
        pass

    async def _handle_interaction_update(self, tab_ref: TabReference) -> None:
        self._active_tab_ref = tab_ref
        await self._clear_list_view()
        await self._clear_table_view()

    async def _handle_content_fetched(
        self,
        tab_ref: TabReference,
        screenshot: Optional[Image.Image],
        scroll_y: Optional[int],
        dom_string: Optional[str],
    ) -> None:
        """Handles callback from monitor after content (HTML, screenshot, DOM) is fetched."""
        html_content = tab_ref.html  # HTML is now part of the TabReference passed

        if not html_content:
            logger.warning(
                f"Monitor Content Fetched (Tab {tab_ref.id}): No HTML content in TabReference."
            )
            await self._clear_list_view()
            await self._clear_table_view()
            return

        logger.info(
            f"Monitor Content Fetched (Tab {tab_ref.id}): HTML length {len(html_content)}, ScrollY: {scroll_y}"
        )
        self._active_tab_ref = tab_ref
        self._active_tab_dom_string = dom_string

        # TODO: Potentially use the screenshot image?
        if screenshot:
            logger.debug(f"Received screenshot of size {screenshot.size}")
            # Example: Save screenshot?
            # try:
            #     screenshot.save(f"./screenshot_{tab_ref.id}.png")
            # except Exception as save_err:
            #     logger.error(f"Failed to save screenshot: {save_err}")

        try:
            page_markdown = markdownify(html_content, heading_style="ATX")
            list_view = self.query_one("#markdown-list", ListView)
            await list_view.clear()
            md_widget = Markdown(page_markdown.strip(), classes="full-page-markdown")
            list_item = ListItem(md_widget, classes="markdown-list-item full-page-item")
            await list_view.append(list_item)
        except Exception as md_err:
            logger.error(f"Failed to update markdown view from monitor callback: {md_err}")

        try:
            table = self.query_one(DataTable)
            table.clear()
            table.add_row(
                html_content[:5000] + "..." if len(html_content) > 5000 else html_content,
                key=f"html_{tab_ref.id}",
            )
        except Exception as table_err:
            logger.error(f"Failed to update data table from monitor callback: {table_err}")

        # vision-based initial description proposal using screenshot
        if screenshot and self._openai_client:
            # ensure client is not None for type checking
            assert self._openai_client is not None
            client = self._openai_client

            if self._vision_worker and self._vision_worker.is_running:
                self._vision_worker.cancel()

            async def _do_vision_proposal():
                try:
                    proposal = await propose_select(client, screenshot)
                    if isinstance(proposal, AutoProposal):
                        desc = proposal.proposed_description

                        def _update_input():
                            try:
                                prompt_input = self.query_one("#prompt-input", Input)
                                prompt_input.value = desc
                            except Exception as e:
                                logger.error(f"Error updating input from vision proposal: {e}")

                        self.app.call_later(_update_input)
                except Exception as e:
                    logger.error(f"Error during vision proposal: {e}", exc_info=True)

            self._vision_worker = self.run_worker(
                _do_vision_proposal(), exclusive=True, group="vision_proposal"
            )

    async def _handle_rehighlight(self) -> None:
        await self.trigger_rehighlight()

    async def action_check_chrome_status(self) -> None:
        home_panel = self.query_one(HomePanel)
        home_panel.status = "checking"
        await asyncio.sleep(0.1)

        new_status: ChromeStatus = "error"
        try:
            is_running = await chrome_launcher.is_chrome_process_running()
            if not is_running:
                new_status = "not_running"
            else:
                debug_active = await chrome_launcher.is_chrome_debug_port_active()
                if not debug_active:
                    new_status = "no_debug_port"
                else:
                    new_status = "ready_to_connect"
        except Exception as e:
            logger.error(f"Error checking Chrome status: {e}", exc_info=True)
            new_status = "error"

        home_panel.status = new_status
        logger.info(f"Chrome status check result: {new_status}")

        # --- Automatically connect if ready ---
        if new_status == "ready_to_connect":
            logger.info("Chrome is ready, automatically attempting to connect monitor...")
            # Use call_later to avoid potential blocking/re-entrancy issues
            # if action_connect_monitor takes time or updates status itself.
            self.app.call_later(self.action_connect_monitor)

    async def action_launch_chrome(self) -> None:
        logger.info("Action: Launching Chrome...")
        home_panel = self.query_one(HomePanel)
        home_panel.status = "checking"
        success = await chrome_launcher.launch_chrome()
        if not success:
            logger.error("Failed to launch Chrome via launcher.")
            home_panel.status = "error"
            return
        await asyncio.sleep(1.0)
        await self.action_check_chrome_status()

    async def action_restart_chrome(self) -> None:
        logger.info("Action: Restarting Chrome...")
        home_panel = self.query_one(HomePanel)
        home_panel.status = "checking"
        success = await chrome_launcher.restart_chrome_with_debug_port()
        if not success:
            logger.error("Failed to restart Chrome via launcher.")
            home_panel.status = "error"
            return
        await asyncio.sleep(1.0)
        await self.action_check_chrome_status()

    async def action_connect_monitor(self) -> None:
        logger.info("Action: Connecting monitor...")
        home_panel = self.query_one(HomePanel)
        home_panel.status = "connecting"
        await asyncio.sleep(0.1)

        if not self._chrome_monitor:
            logger.error("Monitor not initialized, cannot connect.")
            home_panel.status = "error"
            return

        if not await chrome_launcher.is_chrome_debug_port_active():
            logger.error("Debug port became inactive before monitor could start.")
            await self.action_check_chrome_status()
            return

        try:
            success = await self._chrome_monitor.start_monitoring(
                on_polling_change_callback=self._handle_polling_change,
                on_interaction_update_callback=self._handle_interaction_update,
                on_content_fetched_callback=self._handle_content_fetched,
            )
            if success:
                logger.info("Chrome Monitor started successfully.")
                home_panel.status = "connected"
            else:
                logger.error("Failed to start Chrome Monitor.")
                home_panel.status = "error"
        except Exception as e:
            logger.error(f"Error starting Chrome Monitor: {e}", exc_info=True)
            home_panel.status = "error"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "check-chrome-status":
            await self.action_check_chrome_status()
        elif button_id == "launch-chrome":
            await self.action_launch_chrome()
        elif button_id == "restart-chrome":
            await self.action_restart_chrome()
        else:
            logger.warning(f"Unhandled button press: {event.button.id}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt-input":
            target_description = event.value.strip()
            if not target_description:
                return
            if (
                not self._active_tab_ref
                or not self._chrome_monitor
                or not self._chrome_monitor._monitoring
            ):
                logger.warning(
                    "Submit attempted but monitor not connected or no active tab identified."
                )
                return
            event.input.value = ""
            if self._agent_worker and self._agent_worker.is_running:
                logger.info("Cancelling previous agent worker.")
                self._agent_worker.cancel()
            self._agent_worker = self.run_worker(
                self._run_agent_and_highlight(target_description),
                exclusive=True,
                group="agent_worker",
            )

    async def action_quit(self) -> None:
        self.shutdown_event.set()
        if self._chrome_monitor:
            await self._chrome_monitor.stop_monitoring()
        if self._agent_worker and self._agent_worker.is_running:
            self._agent_worker.cancel()
        self._highlighter.set_active(False)
        if self._active_tab_ref:
            try:
                self.call_later(self._highlighter.clear, self._active_tab_ref)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error scheduling highlight clear on exit: {e}")
        self.app.exit()

    def action_open_log_file(self) -> None:
        try:
            log_panel_widget = self.query_one(LogPanel)
            log_panel_widget.open_log_in_editor()
        except Exception as e:
            logger.error(f"Failed to open log file via LogPanel: {e}", exc_info=True)

    def action_toggle_dark(self) -> None:
        if self.theme == THEME_LIGHT:
            self.theme = THEME_DARK
        else:
            self.theme = THEME_LIGHT

    async def _run_agent_and_highlight(self, target_description: str) -> None:
        if not self._active_tab_ref or not self._active_tab_ref.html:
            logger.warning("Cannot run agent: No active tab reference with html.")
            return

        latest_screenshot: Optional[Image.Image] = None
        tab_ref = self._active_tab_ref
        current_html = tab_ref.html
        current_dom_string = self._active_tab_dom_string
        current_url = tab_ref.url

        if not current_html:
            logger.error(
                f"Cannot run agent: HTML content missing in active tab ref {tab_ref.id}. Aborting."
            )
            return
        if not current_url:
            logger.error(f"Cannot run agent: URL missing in active tab ref {tab_ref.id}. Aborting.")
            return

        logger.info(
            f"Running SelectorAgent logic for target '{target_description}' on tab {tab_ref.id}"
        )
        base_url_for_agent = current_url

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
                nonlocal latest_screenshot
                logger.debug(f"Agent calling evaluate_selector: '{selector}'")
                result = await tools_instance.evaluate_selector(
                    selector=selector,
                    target_text_to_check=target_text_to_check,
                    **kwargs,
                    return_matched_html=True,
                )
                if result and not result.error:
                    html_to_show = result.matched_html_snippets or []
                    await _update_list_with_markdown(html_to_show, selector)
                if result and result.element_count > 0 and not result.error:
                    logger.debug(f"Highlighting intermediate selector: '{selector}'")
                    success, img = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="yellow"
                    )
                    if success and img:
                        latest_screenshot = img
                return result

            async def get_children_tags_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot
                logger.debug(f"Agent calling get_children_tags: '{selector}'")
                result = await tools_instance.get_children_tags(selector=selector, **kwargs)
                if result and result.parent_found and not result.error:
                    logger.debug(f"Highlighting parent for get_children_tags: '{selector}'")
                    success, img = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="red"
                    )
                    if success and img:
                        latest_screenshot = img
                return result

            async def get_siblings_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot
                logger.debug(f"Agent calling get_siblings: '{selector}'")
                result = await tools_instance.get_siblings(selector=selector, **kwargs)
                if result and result.element_found and not result.error:
                    success, img = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                    if success and img:
                        latest_screenshot = img
                return result

            async def extract_data_from_element_wrapper(selector: str, **kwargs):
                nonlocal latest_screenshot
                logger.debug(f"Highlighting potentially final selector: '{selector}'")
                success, img = await self._highlighter.highlight(
                    self._active_tab_ref, selector, color="lime"
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

            system_prompt = SELECTOR_PROMPT_BASE
            if current_dom_string:
                system_prompt += SELECTOR_PROMPT_DOM_TEMPLATE.format(
                    dom_representation=current_dom_string
                )
            else:
                logger.warning(f"Proceeding without DOM string representation for tab {tab_ref.id}")

            agent = Agent(
                "openai:gpt-4.1",
                output_type=SelectorProposal,
                tools=wrapped_tools,
                system_prompt=system_prompt,
            )

            logger.info("Starting agent.run()...")
            query_parts = [
                f"Generate the most STABLE CSS selector to target '{target_description}'.",
                "Prioritize stable attributes and classes.",
                "CRITICAL: Your FINAL output MUST be a single JSON object conforming EXACTLY to the SelectorProposal schema. "
                "This JSON object MUST include values for the fields: 'proposed_selector' (string) and 'reasoning' (string). "
                "DO NOT include other fields like 'final_verification' or 'extraction_result' in the final JSON output.",
            ]
            query = " ".join(query_parts)
            agent_input: Any = query
            agent_run_result = await agent.run(agent_input)
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
                self._last_proposed_selector = None
                self.call_later(self._clear_list_view)
        except Exception as e:
            logger.error(
                f"Error running SelectorAgent for target '{target_description}': {e}", exc_info=True
            )
            self._last_proposed_selector = None
            self.call_later(self._clear_list_view)

    async def trigger_rehighlight(self):
        if self._active_tab_ref and self._last_proposed_selector:
            logger.info(
                f"Re-highlighting '{self._last_proposed_selector}' on tab {self._active_tab_ref.id}"
            )
            await self._highlighter.rehighlight(self._active_tab_ref)
        else:
            logger.debug("Re-highlight requested but no active tab or last selector.")

    async def _clear_list_view(self) -> None:
        try:
            list_view = self.query_one("#markdown-list", ListView)
            await list_view.clear()
        except Exception as e:
            logger.error(f"Failed to query or clear list view: {e}")

    async def _clear_table_view(self) -> None:
        try:
            table = self.query_one(DataTable)
            table.clear()
        except Exception as e:
            logger.error(f"Failed to query or clear data table: {e}")


if __name__ == "__main__":
    get_logger("__main__")
    app = SelectronApp()
    app.run()

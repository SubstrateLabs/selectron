import asyncio
from typing import Any, Optional

from PIL import Image
from pydantic_ai import Agent, Tool
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.timer import Timer
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker

from selectron.ai.propose_selection import propose_selection
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
from selectron.util.model_config import ModelConfig

logger = get_logger(__name__)
LOG_PATH = get_app_dir() / "selectron.log"
THEME_DARK = "nord"
THEME_LIGHT = "solarized-light"
DEFAULT_THEME = THEME_LIGHT


class SelectronApp(App[None]):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        Binding(key="ctrl+c", action="quit", description="⣏ Quit ⣹", show=False),
        Binding(key="ctrl+q", action="quit", description="⣏ Quit ⣹", show=True),
        Binding(key="ctrl+t", action="toggle_dark", description="⣏ Light/Dark Mode ⣹", show=True),
        Binding(key="ctrl+l", action="open_log_file", description="⣏ .log file ⣹", show=True),
    ]
    shutdown_event: asyncio.Event
    _active_tab_ref: Optional[TabReference] = None
    _active_tab_dom_string: Optional[str] = None
    _agent_worker: Optional[Worker[None]] = None
    _vision_worker: Optional[Worker[None]] = None
    _highlighter: ChromeHighlighter
    _last_proposed_selector: Optional[str] = None
    _chrome_monitor: Optional[ChromeMonitor] = None
    _vision_proposal_done_for_tab: Optional[str] = None
    _input_debounce_timer: Optional[Timer] = None

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.title = "Selectron"
        self.shutdown_event = asyncio.Event()
        self._highlighter = ChromeHighlighter()
        self._model_config = model_config

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main-container"):
            with TabbedContent(initial="home-tab"):
                with TabPane("⣏ Home ⣹", id="home-tab"):
                    yield HomePanel(id="home-panel-widget")
                with TabPane("⣏ Logs ⣹", id="logs-tab"):
                    yield LogPanel(log_file_path=LOG_PATH, id="log-panel-widget")
                with TabPane("⣏ Parsed Data ⣹", id="table-tab"):
                    yield DataTable(id="data-table")
        with Container(classes="input-bar"):
            yield Button("Start AI selection", id="submit-button")
            yield Input(placeholder="Enter prompt (or let AI propose...)", id="prompt-input")
            yield Label("No active tab (interact to activate)", id="active-tab-url-label")
        yield Footer()

    async def on_mount(self) -> None:
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
        # Reset vision proposal flag for any tab that navigated
        navigated_ids = {new_tab.id for new_tab, _ in event.navigated_tabs}
        if navigated_ids:
            logger.debug(
                f"Polling detected navigation for tabs: {navigated_ids}. Resetting vision flag."
            )
            # We only track one flag, so if *any* navigation occurs, reset it.
            # This assumes the user interaction/focus will align with one of the navigated tabs soon.
            # A more complex approach might involve checking if the *currently active* tab navigated,
            # but let's start simple.
            self._vision_proposal_done_for_tab = None

    async def _handle_interaction_update(self, tab_ref: TabReference) -> None:
        self._active_tab_ref = tab_ref
        try:
            url_label = self.query_one("#active-tab-url-label", Label)
            if tab_ref and tab_ref.url:
                url_label.update(tab_ref.url)
        except Exception as label_err:
            logger.error(f"Failed to update URL label on interaction: {label_err}")

        await self._clear_table_view()
        # Reset vision proposal flag only if the interacted tab is NEW
        current_active_id = self._active_tab_ref.id if self._active_tab_ref else None
        if tab_ref and tab_ref.id != current_active_id:
            logger.debug(
                f"Interaction detected on DIFFERENT tab {tab_ref.id} (current: {current_active_id}). Priming vision proposal."
            )
            # Hide any status badge from previous tab interaction
            await self._highlighter.hide_agent_status(tab_ref)
            self._vision_proposal_done_for_tab = (
                None  # Allow proposal for this tab upon next content fetch
            )

        try:
            input_widget = self.query_one("#prompt-input", Input)
            if input_widget.value:  # Clear input only if it has value
                input_widget.value = ""
                # Also hide the status badge since we cleared the input
                if self._active_tab_ref:
                    await self._highlighter.hide_agent_status(self._active_tab_ref)
            # Clear prompt regardless of tab match
        except Exception as e:
            logger.warning(f"Could not clear prompt input on interaction: {e}")
        # Vision proposal logic moved entirely to _handle_content_fetched

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
            await self._clear_table_view()
            return

        # Always update the active ref and DOM string first
        self._active_tab_ref = tab_ref
        self._active_tab_dom_string = dom_string

        # Update UI Label using the latest fetched info
        try:
            url_label = self.query_one("#active-tab-url-label", Label)
            if tab_ref and tab_ref.url:
                url_label.update(tab_ref.url)
            else:
                url_label.update("No active tab URL")
        except Exception as label_err:
            logger.error(f"Failed to update URL label on content fetch: {label_err}")

        try:
            table = self.query_one(DataTable)
            table.clear()
            # Keep updating the data table
            # Guard against potentially missing html
            html_to_display = "(No HTML content)"
            if html_content:
                html_to_display = (
                    html_content[:5000] + "..." if len(html_content) > 5000 else html_content
                )

            table.add_row(
                html_to_display,
                key=f"html_{tab_ref.id}",
            )
        except Exception as table_err:
            logger.error(f"Failed to update data table from monitor callback: {table_err}")

        # Vision proposal trigger logic (relies on _vision_proposal_done_for_tab flag)
        # Only trigger if screenshot is available AND the flag indicates proposal is needed for this tab
        if (
            screenshot
            and self._active_tab_ref
            and self._vision_proposal_done_for_tab != self._active_tab_ref.id
        ):
            await self._highlighter.show_agent_status(
                self._active_tab_ref, "Proposing selection...", state="thinking"
            )

            if self._vision_worker and self._vision_worker.is_running:
                logger.debug("Cancelling previous vision worker.")
                self._vision_worker.cancel()

            async def _do_vision_proposal():
                try:
                    proposal = await propose_selection(screenshot, self._model_config)
                    # Set the flag *before* updating UI to prevent potential race conditions
                    if self._active_tab_ref:
                        self._vision_proposal_done_for_tab = self._active_tab_ref.id

                    if isinstance(proposal, AutoProposal):
                        desc = proposal.proposed_description

                        def _update_input():
                            try:
                                prompt_input = self.query_one("#prompt-input", Input)
                                prompt_input.value = desc
                                # Show "Ready" badge after AI proposal fills input
                                if desc and self._active_tab_ref:
                                    self.app.call_later(
                                        self._highlighter.show_agent_status,
                                        self._active_tab_ref,
                                        desc,
                                        state="idle",
                                    )
                            except Exception as e:
                                logger.error(f"Error updating input from vision proposal: {e}")

                        self.app.call_later(_update_input)

                        # Update badge to "Ready" after successful proposal
                        if self._active_tab_ref:
                            self.app.call_later(
                                self._highlighter.show_agent_status,
                                self._active_tab_ref,
                                desc,
                                state="idle",
                            )

                except Exception as e:
                    logger.error(f"Error during vision proposal: {e}", exc_info=True)
                    # Hide status badge on vision proposal error
                    if self._active_tab_ref:
                        self.app.call_later(
                            lambda: self._highlighter.hide_agent_status(self._active_tab_ref)
                        )
                    # Hide status badge on error? Maybe not, input might still be valid.
                    # Potentially reset the flag on error? Or leave it set to prevent retries?
                    # Let's leave it set for now.

            self._vision_worker = self.run_worker(
                _do_vision_proposal(), exclusive=True, group="vision_proposal"
            )
        elif (
            screenshot
            and self._active_tab_ref
            and self._vision_proposal_done_for_tab == self._active_tab_ref.id
        ):
            pass
        elif not screenshot:
            logger.debug("Skipping vision proposal: No screenshot available.")

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
        if new_status == "ready_to_connect":
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
        elif button_id == "submit-button":
            input_widget = self.query_one("#prompt-input", Input)
            await self.on_input_submitted(Input.Submitted(input_widget, input_widget.value))
        else:
            logger.warning(f"Unhandled button press: {event.button.id}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt-input":
            # When submitted, the agent will start and show its own status, overriding "Ready"
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
            logger.info("Cancelling agent worker on quit.")
            self._agent_worker.cancel()
            # Ensure badge is hidden on quit, even if agent was running
            if self._active_tab_ref:
                await self._highlighter.hide_agent_status(self._active_tab_ref)
        self._highlighter.set_active(False)
        if self._active_tab_ref:
            try:
                self.call_later(self._highlighter.clear, self._active_tab_ref)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error scheduling highlight clear on exit: {e}")
        # Reset URL label on quit
        try:
            url_label = self.query_one("#active-tab-url-label", Label)
            url_label.update("No active tab (interact to activate)")
        except Exception as label_err:
            logger.warning(f"Failed to reset URL label on quit: {label_err}")
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

        tab_ref = self._active_tab_ref
        current_html = tab_ref.html
        current_dom_string = self._active_tab_dom_string
        current_url = tab_ref.url

        if not current_html:
            logger.error(
                f"Cannot run agent: HTML content missing in active tab ref {tab_ref.id}. Aborting."
            )
            await self._highlighter.hide_agent_status(tab_ref)  # Hide badge on abort
            return
        if not current_url:
            logger.error(f"Cannot run agent: URL missing in active tab ref {tab_ref.id}. Aborting.")
            await self._highlighter.hide_agent_status(tab_ref)  # Hide badge on abort
            return

        logger.info(
            f"Running SelectorAgent logic for target '{target_description}' on tab {tab_ref.id}"
        )
        base_url_for_agent = current_url

        try:
            # --- Initial status update ---
            await self._highlighter.show_agent_status(
                tab_ref, "Agent starting...", state="thinking"
            )
            # Use the same executor for subsequent operations if created by highlighter
            # Note: This assumes show_agent_status might create one we can reuse.
            # A more robust way might be to explicitly create one here and pass it.
            # Let's assume _execute_js_on_tab handles creation/reuse implicitly for now.

            tools_instance = SelectorTools(html_content=current_html, base_url=base_url_for_agent)

            async def evaluate_selector_wrapper(selector: str, target_text_to_check: str, **kwargs):
                await self._highlighter.show_agent_status(
                    tab_ref, f"Tool: evaluate_selector('{selector[:30]}...')", state="sending"
                )
                result = await tools_instance.evaluate_selector(
                    selector=selector,
                    target_text_to_check=target_text_to_check,
                    **kwargs,
                    return_matched_html=True,
                )
                if result and result.element_count > 0 and not result.error:
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="yellow"
                    )
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: evaluate_selector OK ({result.element_count} found)",
                        state="received_success",
                    )
                    if success:
                        pass
                elif result and result.error:
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: evaluate_selector Error: {result.error[:50]}...",
                        state="received_error",
                    )
                else:
                    # No elements found or other non-error case
                    await self._highlighter.show_agent_status(
                        tab_ref, "Tool: evaluate_selector OK (0 found)", state="received_success"
                    )  # Still success technically
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="yellow"
                    )
                return result

            async def get_children_tags_wrapper(selector: str, **kwargs):
                await self._highlighter.show_agent_status(
                    tab_ref, f"Tool: get_children_tags('{selector[:30]}...')", state="sending"
                )
                result = await tools_instance.get_children_tags(selector=selector, **kwargs)
                if result and result.parent_found and not result.error:
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="red"
                    )
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: get_children_tags OK ({len(result.children_details or [])} children)",
                        state="received_success",
                    )
                    if success:
                        pass
                elif result and result.error:
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: get_children_tags Error: {result.error[:50]}...",
                        state="received_error",
                    )
                else:
                    # Parent not found or other non-error case
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        "Tool: get_children_tags OK (Parent not found)",
                        state="received_success",
                    )
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="red"
                    )
                return result

            async def get_siblings_wrapper(selector: str, **kwargs):
                await self._highlighter.show_agent_status(
                    tab_ref, f"Tool: get_siblings('{selector[:30]}...')", state="sending"
                )
                result = await tools_instance.get_siblings(selector=selector, **kwargs)
                if result and result.element_found and not result.error:
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: get_siblings OK ({len(result.siblings or [])} siblings)",
                        state="received_success",
                    )
                    if success:
                        pass
                elif result and result.error:
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: get_siblings Error: {result.error[:50]}...",
                        state="received_error",
                    )
                else:
                    # Element not found or other non-error case
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        "Tool: get_siblings OK (Element not found)",
                        state="received_success",
                    )
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                return result

            async def extract_data_from_element_wrapper(selector: str, **kwargs):
                await self._highlighter.show_agent_status(
                    tab_ref,
                    f"Tool: extract_data_from_element('{selector[:30]}...')",
                    state="sending",
                )
                # No highlight here anymore, final highlight happens after agent completion.
                result = await tools_instance.extract_data_from_element(selector=selector, **kwargs)
                # Check for error first. If no error, extraction was attempted (element likely found).
                if result and not result.error:
                    # Access extracted_data directly, it's a dict - NO, check individual fields
                    # Check which fields have data
                    extracted_count = sum(
                        1
                        for val in [
                            result.extracted_text,
                            result.extracted_attribute_value,
                            result.extracted_markdown,
                            result.extracted_html,
                        ]
                        if val is not None
                    )
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: extract_data OK ({extracted_count} fields populated)",
                        state="received_success",
                    )
                elif result and result.error:
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        f"Tool: extract_data Error: {result.error[:50]}...",
                        state="received_error",
                    )
                else:
                    await self._highlighter.show_agent_status(
                        tab_ref,
                        "Tool: extract_data OK (Element not found or no data)",
                        state="received_success",
                    )
                return result

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

            # Update status before agent run
            await self._highlighter.show_agent_status(tab_ref, "Thinking...", state="thinking")

            agent = Agent(
                self._model_config.selector_agent_model,
                output_type=SelectorProposal,
                tools=wrapped_tools,
                system_prompt=system_prompt,
            )
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

            # --- Status update after agent run ---
            if isinstance(agent_run_result.output, SelectorProposal):
                proposal = agent_run_result.output
                logger.info(
                    f"FINISHED. Proposal: {proposal.proposed_selector}\\nREASONING: {proposal.reasoning}"
                )
                await self._highlighter.show_agent_status(
                    tab_ref,
                    "Success",
                    state="final_success",
                )
                self._last_proposed_selector = proposal.proposed_selector
                # Highlight the final proposed selector in lime
                success = await self._highlighter.highlight(
                    self._active_tab_ref, proposal.proposed_selector, color="lime"
                )
                # Keep success badge briefly, then hide? Or hide immediately? Let's hide after a short delay.
                self.app.call_later(self._delayed_hide_status)

                if success:
                    pass
                elif not success:
                    logger.warning(
                        f"Final highlight failed for selector: '{proposal.proposed_selector}'"
                    )
            else:
                logger.error(
                    f"Agent returned unexpected output type: {type(agent_run_result.output)}"
                )
                await self._highlighter.show_agent_status(
                    tab_ref,
                    "Agent Error: Unexpected output type",
                    state="received_error",
                )
                self._last_proposed_selector = None
                self.call_later(self._clear_table_view)
                self.app.call_later(self._delayed_hide_status)  # Hide error badge too
        except Exception as e:
            logger.error(
                f"Error running SelectorAgent for target '{target_description}': {e}", exc_info=True
            )
            # --- Status update on agent exception ---
            if tab_ref:  # Check if tab_ref is still valid
                error_msg = f"Agent Error: {type(e).__name__}"
                await self._highlighter.show_agent_status(
                    tab_ref, error_msg, state="received_error"
                )
                if self.app:  # Ensure app context is available
                    self.app.call_later(self._delayed_hide_status)

            self._last_proposed_selector = None
            self.call_later(self._clear_table_view)
        finally:
            # Ensure the badge is eventually hidden if not handled by success/error paths with delays
            # This is a fallback in case the worker is cancelled or ends unexpectedly
            if self._active_tab_ref:
                # Let the delayed hides handle it. If cancelled, action_quit handles it.
                pass

    async def trigger_rehighlight(self):
        # Check if there's an active tab and if the highlighter state indicates highlights are active
        if self._active_tab_ref and self._highlighter.is_active():
            await self._highlighter.rehighlight(self._active_tab_ref)

    async def _clear_table_view(self) -> None:
        try:
            table = self.query_one(DataTable)
            table.clear()
        except Exception as e:
            logger.error(f"Failed to query or clear data table: {e}")

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle changes in the prompt input using a timer for debouncing."""
        if event.input.id == "prompt-input":
            # Cancel existing timer if it exists
            if self._input_debounce_timer:
                self._input_debounce_timer.stop()

            # Define the action to perform after debounce timeout
            async def _update_status_after_debounce():
                current_value = event.value.strip()  # Use event value captured by closure
                if self._active_tab_ref:
                    if current_value:
                        # Display the current input value in the badge
                        await self._highlighter.show_agent_status(
                            self._active_tab_ref, current_value, state="idle"
                        )
                    else:
                        await self._highlighter.hide_agent_status(self._active_tab_ref)
                self._input_debounce_timer = None  # Clear timer ref after execution

            # Start a new timer
            self._input_debounce_timer = self.set_timer(
                0.5, _update_status_after_debounce, name="input_debounce"
            )

    async def _delayed_hide_status(self) -> None:
        """Helper method called via call_later to hide the status badge after a delay."""
        await asyncio.sleep(3.0)  # Handle the delay internally
        if self._active_tab_ref:
            logger.debug("Hiding agent status badge after delay.")
            await self._highlighter.hide_agent_status(self._active_tab_ref)

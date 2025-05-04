import asyncio
import os
from typing import Any, Optional

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

from selectron.ai.selector_prompt import (
    SELECTOR_PROMPT_BASE,
    SELECTOR_PROMPT_DOM_TEMPLATE,
)
from selectron.ai.selector_tools import (
    SelectorTools,
)
from selectron.ai.types import (
    SelectorProposal,
)
from selectron.chrome import chrome_launcher
from selectron.chrome.chrome_highlighter import ChromeHighlighter
from selectron.chrome.chrome_monitor import ChromeMonitor
from selectron.chrome.types import TabReference
from selectron.cli.home_panel import ChromeStatus, HomePanel
from selectron.cli.log_panel import LogPanel
from selectron.cli.monitor_handler import MonitorEventHandler
from selectron.util.debug_helpers import save_debug_elements
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig

logger = get_logger(__name__)
LOG_PATH = get_app_dir() / "selectron.log"
THEME_DARK = "nord"
THEME_LIGHT = "solarized-light"
DEFAULT_THEME = THEME_LIGHT


class SelectronApp(App[None]):
    _debug_write_selection: bool = os.getenv("SLT_DBG_WRITE_SELECTION", "false").lower() == "true"
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
    _propose_selection_worker: Optional[Worker[None]] = None
    _highlighter: ChromeHighlighter
    _last_proposed_selector: Optional[str] = None
    _chrome_monitor: Optional[ChromeMonitor] = None
    _propose_selection_done_for_tab: Optional[str] = None
    _input_debounce_timer: Optional[Timer] = None
    _monitor_handler: Optional[MonitorEventHandler] = None

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
                with TabPane("⣏ Parsed Data ⣹", id="table-tab"):
                    yield DataTable(id="data-table")
                with TabPane("⣏ Logs ⣹", id="logs-tab"):
                    yield LogPanel(log_file_path=LOG_PATH, id="log-panel-widget")
        with Container(classes="input-bar"):
            with Container(id="button-row", classes="button-status-row"):
                yield Button("Start AI selection", id="submit-button")
                yield Label("", id="agent-status-label")  # New agent status label
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

        # Instantiate MonitorEventHandler after widgets are potentially available
        try:
            url_label = self.query_one("#active-tab-url-label", Label)
            data_table = self.query_one(DataTable)
            prompt_input = self.query_one("#prompt-input", Input)
            self._monitor_handler = MonitorEventHandler(
                app=self,
                highlighter=self._highlighter,
                url_label=url_label,
                data_table=data_table,
                prompt_input=prompt_input,
            )
        except Exception as handler_init_err:
            logger.error(
                f"Failed to initialize MonitorEventHandler: {handler_init_err}", exc_info=True
            )
            # App might be in a bad state here, maybe exit or show error?

        self._chrome_monitor = ChromeMonitor(
            rehighlight_callback=self._handle_rehighlight,
            check_interval=2.0,
            interaction_debounce=0.7,
        )

        await self.action_check_chrome_status()

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
            # Ensure handler is instantiated before starting monitor
            if not self._monitor_handler:
                logger.error("MonitorEventHandler not initialized, cannot start monitor.")
                home_panel.status = "error"
                return

            success = await self._chrome_monitor.start_monitoring(
                on_polling_change_callback=self._monitor_handler.handle_polling_change,  # Use handler method
                on_interaction_update_callback=self._monitor_handler.handle_interaction_update,  # Use handler method
                on_content_fetched_callback=self._monitor_handler.handle_content_fetched,  # Use handler method
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
            # Clear previous highlights before starting a new selection
            if self._active_tab_ref:
                await self._highlighter.clear(self._active_tab_ref)

            if (
                not self._active_tab_ref
                or not self._chrome_monitor
                or not self._chrome_monitor._monitoring
            ):
                logger.warning(
                    "Submit attempted but monitor not connected or no active tab identified."
                )
                # Optionally update UI status here if desired
                # await self._update_ui_status("Error: Not connected", state="received_error")
                return

            # Update UI status immediately to clear any debounce text
            await self._update_ui_status("Preparing agent...", state="thinking")

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
            # Badge hiding is handled below, ensuring it happens even if agent wasn't running

        self._highlighter.set_active(False)
        if self._active_tab_ref:
            try:
                # Schedule badge hide AND highlight clear on quit via call_later
                self.call_later(self._highlighter.hide_agent_status, self._active_tab_ref)
                self.call_later(self._highlighter.clear, self._active_tab_ref)
                await asyncio.sleep(0.1)  # Brief pause for calls to potentially start
            except Exception as e:
                logger.warning(f"Error scheduling highlight clear on exit: {e}")
        # Reset URL label on quit
        try:
            url_label = self.query_one("#active-tab-url-label", Label)
            url_label.update("No active tab (interact to activate)")
        except Exception as label_err:
            logger.warning(f"Failed to reset URL label on quit: {label_err}")
        # Reset button state on quit
        try:
            submit_button = self.query_one("#submit-button", Button)
            submit_button.label = "Start AI selection"
            submit_button.disabled = False
        except Exception as button_err:
            logger.warning(f"Failed to reset submit button on quit: {button_err}")

        self.app.exit()

    async def _update_ui_status(self, message: str, state: str, show_spinner: bool = False) -> None:
        """Helper to update both the terminal label and the browser badge."""
        try:
            # Update terminal AGENT STATUS label
            status_label = self.query_one("#agent-status-label", Label)
            status_label.update(message)
        except Exception as e:
            logger.error(f"Failed to update status label: {e}", exc_info=True)

        # Update browser badge (if active tab exists)
        if self._active_tab_ref:
            try:
                await self._highlighter.show_agent_status(
                    self._active_tab_ref, message, state=state, show_spinner=show_spinner
                )
            except Exception as e:
                logger.error(f"Failed to show agent status badge: {e}", exc_info=True)
        else:
            logger.debug(
                f"Skipping browser badge update for status '{message}' (no active tab ref)."
            )

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
            await self._update_ui_status(
                "Agent Error: Missing HTML", state="received_error", show_spinner=False
            )
            return

        tab_ref = self._active_tab_ref
        current_html = tab_ref.html
        current_dom_string = self._active_tab_dom_string
        current_url = tab_ref.url

        # --- Get UI elements --- Needed for status updates and button disabling
        try:
            submit_button = self.query_one("#submit-button", Button)
        except Exception as e:
            logger.error(f"Failed to query submit button: {e}", exc_info=True)
            submit_button = None  # Proceed without button control if query fails

        # --- Disable button and show initial status --- BEFORE the main try block
        if submit_button:
            submit_button.label = "Running AI..."
            submit_button.disabled = True

        # Use the new helper for initial status
        await self._update_ui_status("Agent starting...", state="thinking", show_spinner=True)

        if not current_html:
            logger.error(
                f"Cannot run agent: HTML content missing in active tab ref {tab_ref.id}. Aborting."
            )
            await self._update_ui_status(
                "Agent Error: Missing HTML", state="received_error", show_spinner=False
            )
            # No need to hide badge explicitly, finally block handles button reset
            return
        if not current_url:
            logger.error(f"Cannot run agent: URL missing in active tab ref {tab_ref.id}. Aborting.")
            await self._update_ui_status(
                "Agent Error: Missing URL", state="received_error", show_spinner=False
            )
            return

        tool_call_count = 0
        logger.info(
            f"Running SelectorAgent logic for target '{target_description}' on tab {tab_ref.id}"
        )
        base_url_for_agent = current_url
        try:
            tools_instance = SelectorTools(html_content=current_html, base_url=base_url_for_agent)

            async def evaluate_selector_wrapper(selector: str, target_text_to_check: str, **kwargs):
                nonlocal tool_call_count
                tool_call_count += 1
                status_prefix = f"Tool #{tool_call_count} |"
                await self._update_ui_status(
                    f"{status_prefix} evaluate_selector('{selector[:30]}...')",
                    state="sending",
                    show_spinner=True,
                )
                known_args_for_tool = {
                    "anchor_selector": kwargs.get("anchor_selector"),
                    "max_html_length": kwargs.get("max_html_length"),
                    "max_matches_to_detail": kwargs.get("max_matches_to_detail", None),
                    "return_matched_html": True,  # NOTE: hardcoded?
                }
                filtered_args_for_tool = {
                    k: v for k, v in known_args_for_tool.items() if v is not None
                }
                result = await tools_instance.evaluate_selector(
                    selector=selector,
                    target_text_to_check=target_text_to_check,
                    **filtered_args_for_tool,  # Pass only the filtered, known arguments
                )
                if result and result.element_count > 0 and not result.error:
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="yellow"
                    )
                    await self._update_ui_status(
                        f"{status_prefix} evaluate_selector OK ({result.element_count} found)",
                        state="received_success",
                        show_spinner=True,
                    )
                    if success:
                        pass
                elif result and result.element_count == 0 and not result.error:
                    await self._update_ui_status(
                        f"{status_prefix} Selector found 0 elements",
                        state="received_no_results",  # Orange: 0 found
                        show_spinner=True,
                    )
                    # Still try to highlight (might clear previous) even if 0 found now
                    await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="yellow"
                    )
                elif result and result.error:
                    await self._update_ui_status(
                        f"{status_prefix} evaluate_selector Error: {result.error[:50]}...",
                        state="received_error",  # Salmon: Error
                        show_spinner=True,
                    )
                # NOTE: Implicit else (result is None) - should not happen
                return result

            async def get_children_tags_wrapper(selector: str, **kwargs):
                nonlocal tool_call_count
                tool_call_count += 1
                status_prefix = f"[Tool #{tool_call_count}]"
                await self._update_ui_status(
                    f"{status_prefix} get_children_tags('{selector[:30]}...')",
                    state="sending",
                    show_spinner=True,
                )
                known_args_for_tool = {
                    "anchor_selector": kwargs.get("anchor_selector"),
                }
                filtered_args_for_tool = {
                    k: v for k, v in known_args_for_tool.items() if v is not None
                }
                result = await tools_instance.get_children_tags(
                    selector=selector, **filtered_args_for_tool
                )
                if result and result.parent_found and not result.error:
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="red"
                    )
                    await self._update_ui_status(
                        f"{status_prefix} get_children_tags OK ({len(result.children_details or [])} children)",
                        state="received_success",
                        show_spinner=True,
                    )
                    if success:
                        pass
                elif result and not result.parent_found and not result.error:
                    await self._update_ui_status(
                        f"{status_prefix} Parent selector found 0 elements",
                        state="received_no_results",
                        show_spinner=True,
                    )
                    # NOTE: Still highlight the selector even if parent not found?
                    await self._highlighter.highlight(self._active_tab_ref, selector, color="red")
                elif result and result.error:
                    await self._update_ui_status(
                        f"{status_prefix} get_children_tags Error: {result.error[:50]}...",
                        state="received_error",
                        show_spinner=True,
                    )
                # NOTE: Implicit else (result is None)
                return result

            async def get_siblings_wrapper(selector: str, **kwargs):
                nonlocal tool_call_count
                tool_call_count += 1
                status_prefix = f"[Tool #{tool_call_count}]"
                await self._update_ui_status(
                    f"{status_prefix} get_siblings('{selector[:30]}...')",
                    state="sending",
                    show_spinner=True,
                )
                known_args_for_tool = {
                    "anchor_selector": kwargs.get("anchor_selector"),
                }
                filtered_args_for_tool = {
                    k: v for k, v in known_args_for_tool.items() if v is not None
                }
                result = await tools_instance.get_siblings(
                    selector=selector, **filtered_args_for_tool
                )
                if result and result.element_found and not result.error:
                    success = await self._highlighter.highlight(
                        self._active_tab_ref, selector, color="blue"
                    )
                    await self._update_ui_status(
                        f"{status_prefix} get_siblings OK ({len(result.siblings or [])} siblings)",
                        state="received_success",
                        show_spinner=True,
                    )
                    if success:
                        pass
                elif result and not result.element_found and not result.error:
                    await self._update_ui_status(
                        f"{status_prefix} Element selector found 0 elements",
                        state="received_no_results",
                        show_spinner=True,
                    )
                    # NOTE: Still highlight the selector even if element not found?
                    await self._highlighter.highlight(self._active_tab_ref, selector, color="blue")
                elif result and result.error:
                    await self._update_ui_status(
                        f"{status_prefix} get_siblings Error: {result.error[:50]}...",
                        state="received_error",
                        show_spinner=True,
                    )
                # NOTE: Implicit else (result is None)
                return result

            async def extract_data_from_element_wrapper(selector: str, **kwargs):
                nonlocal tool_call_count
                tool_call_count += 1
                status_prefix = f"[Tool #{tool_call_count}]"
                await self._update_ui_status(
                    f"{status_prefix} extract_data_from_element('{selector[:30]}...')",
                    state="sending",
                    show_spinner=True,
                )
                known_args_for_tool = {
                    "attribute_to_extract": kwargs.get("attribute_to_extract"),
                    "extract_text": kwargs.get(
                        "extract_text", False
                    ),  # NOTE: default false if LLM omits
                    "anchor_selector": kwargs.get("anchor_selector"),
                }
                filtered_args_for_tool = {
                    k: v for k, v in known_args_for_tool.items() if v is not None
                }
                # NOTE: No highlight here anymore, final highlight happens after agent completion.
                result = await tools_instance.extract_data_from_element(
                    selector=selector, **filtered_args_for_tool
                )
                # Check for error first. If no error, extraction was attempted (element likely found).
                if result and not result.error:
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
                    if extracted_count > 0:
                        await self._update_ui_status(
                            f"{status_prefix} extract_data OK ({extracted_count} fields populated)",
                            state="received_success",
                            show_spinner=True,
                        )
                    else:
                        await self._update_ui_status(
                            f"{status_prefix} extract_data OK (No specific data extracted)",
                            state="received_no_results",
                            show_spinner=True,
                        )
                elif result and result.error:
                    await self._update_ui_status(
                        f"{status_prefix} extract_data Error: {result.error[:50]}...",
                        state="received_error",
                        show_spinner=True,
                    )
                # NOTE: Implicit else (result is None)
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

            await self._update_ui_status("Thinking...", state="thinking", show_spinner=True)
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
                    f"FINISHED. Proposal: {proposal.proposed_selector}\nREASONING: {proposal.reasoning}"
                )
                await self._update_ui_status(
                    "Done",
                    state="final_success",
                    show_spinner=False,
                )
                self._last_proposed_selector = proposal.proposed_selector

                # --- DEBUG: Write selected HTML to JSON using the utility function ---
                if self._debug_write_selection:
                    await save_debug_elements(
                        tools_instance=tools_instance,
                        selector=proposal.proposed_selector,
                        target_description=target_description,
                        url=current_url,
                        reasoning=proposal.reasoning,
                    )
                # --- End DEBUG section ---

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
                await self._update_ui_status(
                    "Agent Error: Unexpected output type",
                    state="received_error",
                    show_spinner=False,
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
                await self._update_ui_status(error_msg, state="received_error", show_spinner=False)
                if self.app:  # Ensure app context is available
                    self.app.call_later(self._delayed_hide_status)

            self._last_proposed_selector = None
            self.call_later(self._clear_table_view)
        finally:
            # --- Reset button state --- ALWAYS reset the button in finally
            if submit_button:
                submit_button.label = "Start AI selection"
                submit_button.disabled = False
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
                        # Display the current input value in the badge AND label
                        await self._update_ui_status(current_value, state="idle")
                    # else: # What to do if input is cleared? Revert to default?
                    #     await self._update_ui_status("No active tab (interact to activate)", state="idle")
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
        # Also reset the terminal AGENT STATUS label
        try:
            status_label = self.query_one("#agent-status-label", Label)
            status_label.update("")  # Reset agent status label to empty
        except Exception as e:
            logger.warning(f"Failed to reset status label after delay: {e}")

import asyncio
import os
from typing import Optional

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

from selectron.ai.selector_agent import (
    Highlighter as HighlighterProtocol,
)
from selectron.ai.selector_agent import (
    SelectorAgent,
    SelectorAgentError,
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
                on_polling_change_callback=self._monitor_handler.handle_polling_change,
                on_interaction_update_callback=self._monitor_handler.handle_interaction_update,
                on_content_fetched_callback=self._monitor_handler.handle_content_fetched,
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
                await self._update_ui_status("Error: Not connected", state="received_error")
                return

            # Clear previous highlights before starting a new agent run for this tab
            await self._highlighter.clear(self._active_tab_ref)

            # Update UI status immediately
            await self._update_ui_status("Preparing agent...", state="thinking")

            if self._agent_worker and self._agent_worker.is_running:
                logger.info("Cancelling previous agent worker.")
                self._agent_worker.cancel()

            self._agent_worker = self.run_worker(
                self._run_agent_worker(target_description),
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

    async def _run_agent_worker(self, target_description: str) -> None:
        """Worker task to run the SelectorAgent and handle UI updates."""
        if not self._active_tab_ref or not self._active_tab_ref.html:
            logger.warning("Cannot run agent worker: No active tab reference with html.")
            await self._update_ui_status(
                "Agent Error: Missing HTML", state="received_error", show_spinner=False
            )
            return

        tab_ref = self._active_tab_ref
        current_html = tab_ref.html
        current_dom_string = self._active_tab_dom_string
        current_url = tab_ref.url

        # --- Get UI elements --- Needed for button disabling/enabling
        try:
            submit_button = self.query_one("#submit-button", Button)
        except Exception as e:
            logger.error(f"Failed to query submit button: {e}", exc_info=True)
            submit_button = None

        # --- Disable button --- BEFORE the main try block
        if submit_button:
            submit_button.label = "Running AI..."
            submit_button.disabled = True

        # --- Create Status Callback (Lambda) --- #
        async def status_callback(message: str, state: str, show_spinner: bool):
            await self._update_ui_status(message, state, show_spinner)

        # --- Create Highlighter Adapter --- #
        highlighter_adapter = self._ChromeHighlighterAdapter(self._highlighter, tab_ref)

        # --- Check for essential data before creating agent --- #
        if current_html is None:
            logger.error("Cannot run agent worker: HTML content is missing in tab ref.")
            await self._update_ui_status(
                "Agent Error: Missing HTML", state="received_error", show_spinner=False
            )
            # Need to re-enable button in this error case before returning
            if submit_button:
                submit_button.label = "Start AI selection"
                submit_button.disabled = False
            return
        if current_url is None:
            logger.error("Cannot run agent worker: URL is missing in tab ref.")
            await self._update_ui_status(
                "Agent Error: Missing URL", state="received_error", show_spinner=False
            )
            if submit_button:
                submit_button.label = "Start AI selection"
                submit_button.disabled = False
            return

        proposal: Optional[SelectorProposal] = None
        try:
            # --- Instantiate and run the agent --- #
            agent = SelectorAgent(
                html_content=current_html,
                dom_string=current_dom_string,
                base_url=current_url,
                model_cfg=self._model_config,
                status_cb=status_callback,
                highlighter=highlighter_adapter,
                debug_dump=self._debug_write_selection,
            )

            logger.info(
                f"Running SelectorAgent for target '{target_description}' on tab {tab_ref.id}"
            )
            proposal = await agent.run(target_description)

            # --- Handle Successful Proposal --- #
            if proposal:
                logger.info(f"Worker FINISHED. Proposal: {proposal.proposed_selector}")
                await self._update_ui_status(
                    "Done",
                    state="final_success",
                    show_spinner=False,
                )
                self._last_proposed_selector = proposal.proposed_selector

                # Final highlight with the concrete highlighter
                success = await self._highlighter.highlight(
                    self._active_tab_ref, proposal.proposed_selector, color="lime"
                )
                if not success:
                    logger.warning(
                        f"Final highlight failed for selector: '{proposal.proposed_selector}'"
                    )
                # Schedule badge hide after success
                self.app.call_later(self._delayed_hide_status)

        except SelectorAgentError as agent_err:
            # Agent already logged the specific error and updated status via callback
            logger.error(f"SelectorAgent failed: {agent_err}")
            self._last_proposed_selector = None
            self.call_later(self._clear_table_view)
            self.app.call_later(self._delayed_hide_status)  # Schedule hide for error badge too
        except Exception as e:
            # Catch unexpected errors *outside* the agent's known failure modes
            logger.error(
                f"Unexpected error in worker task for target '{target_description}': {e}",
                exc_info=True,
            )
            error_msg = f"Worker Error: {type(e).__name__}"
            await self._update_ui_status(error_msg, state="received_error", show_spinner=False)
            self._last_proposed_selector = None
            self.call_later(self._clear_table_view)
            self.app.call_later(self._delayed_hide_status)
        finally:
            # --- Re-enable button --- #
            if submit_button:
                submit_button.label = "Start AI selection"
                submit_button.disabled = False
            # Note: Badge hiding is handled by success/error paths scheduling _delayed_hide_status
            # Cancellation is handled by action_quit

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
            if self._input_debounce_timer:
                self._input_debounce_timer.stop()

            async def _update_status_after_debounce():
                current_value = event.value.strip()
                if self._active_tab_ref:
                    if current_value:
                        # Use the concrete highlighter for idle badge updates
                        await self._highlighter.show_agent_status(
                            self._active_tab_ref, current_value, state="idle", show_spinner=False
                        )
                    # Optionally handle clearing the input (e.g., hide badge or show default)
                    # else:
                    #    await self._highlighter.hide_agent_status(self._active_tab_ref)
                self._input_debounce_timer = None

            self._input_debounce_timer = self.set_timer(
                0.5, _update_status_after_debounce, name="input_debounce"
            )

    async def _delayed_hide_status(self) -> None:
        """Helper method called via call_later to hide the status badge after a delay."""
        await asyncio.sleep(3.0)
        if self._active_tab_ref:
            logger.debug("Hiding agent status badge after delay.")
            # Use the concrete highlighter
            await self._highlighter.hide_agent_status(self._active_tab_ref)
        try:
            status_label = self.query_one("#agent-status-label", Label)
            status_label.update("")
        except Exception as e:
            logger.warning(f"Failed to reset status label after delay: {e}")

    # --- Highlighter Adapter --- #
    class _ChromeHighlighterAdapter(HighlighterProtocol):
        """Adapts ChromeHighlighter to the Highlighter protocol for a specific tab."""

        def __init__(self, chrome_highlighter: ChromeHighlighter, tab_ref: TabReference):
            self._highlighter = chrome_highlighter
            self._tab_ref = tab_ref

        async def highlight(self, selector: str, color: str) -> bool:
            # Pass executor=None, let the highlighter manage it
            return await self._highlighter.highlight(self._tab_ref, selector, color)

        async def clear(self) -> None:
            # Pass executor=None
            await self._highlighter.clear(self._tab_ref)

        async def show_agent_status(self, text: str, state: str, show_spinner: bool) -> None:
            # Pass executor=None
            await self._highlighter.show_agent_status(self._tab_ref, text, state, show_spinner)

        async def hide_agent_status(self) -> None:
            # Pass executor=None
            await self._highlighter.hide_agent_status(self._tab_ref)

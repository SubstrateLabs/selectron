import asyncio
import os
import webbrowser
from typing import Literal, Optional

# Add duckdb import
import duckdb
from pydantic_ai import UnexpectedModelBehavior, capture_run_messages
from rich.markup import escape
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

from selectron.ai.codegen_agent import CodegenAgent
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
from selectron.cli.settings_panel import SettingsPanel
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig

logger = get_logger(__name__)
LOG_PATH = get_app_dir() / "selectron.log"
# THEME_DARK = "tokyo-night"
THEME_DARK = "catppuccin-mocha"
THEME_LIGHT = "solarized-light"
DEFAULT_THEME = THEME_LIGHT

AiStatus = Literal["enabled_anthropic", "enabled_openai", "disabled"]


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
    _codegen_worker: Optional[Worker[None]] = None
    _highlighter: ChromeHighlighter
    _last_proposed_selector: Optional[str] = None
    _chrome_monitor: Optional[ChromeMonitor] = None
    _propose_selection_done_for_tab: Optional[str] = None
    _input_debounce_timer: Optional[Timer] = None
    _monitor_handler: Optional[MonitorEventHandler] = None
    _duckdb_ui_conn: Optional[duckdb.DuckDBPyConnection] = (
        None  # ADDED: Store connection for DuckDB UI
    )
    _model_config: ModelConfig
    _ai_status: AiStatus

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.title = "Selectron"
        self.shutdown_event = asyncio.Event()
        self._highlighter = ChromeHighlighter()
        self._model_config = model_config
        self._ai_status = self._determine_ai_status(model_config)

    def _determine_ai_status(self, config: ModelConfig) -> AiStatus:
        if config.provider == "anthropic":
            return "enabled_anthropic"
        elif config.provider == "openai":
            return "enabled_openai"
        else:
            return "disabled"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main-container"):
            with TabbedContent(initial="home-tab"):
                with TabPane("⣏ Home ⣹", id="home-tab"):
                    yield HomePanel(id="home-panel-widget")
                    yield DataTable(id="data-table")
                with TabPane("⣏ Logs ⣹", id="logs-tab"):
                    yield LogPanel(log_file_path=LOG_PATH, id="log-panel-widget")
                with TabPane("⣏ Settings ⣹", id="settings-tab"):
                    yield SettingsPanel(id="settings-panel-widget")
        with Container(classes="input-bar"):
            with Container(id="button-row", classes="button-status-row"):
                yield Button("Start AI selection", id="submit-button")
                yield Button(
                    "Start AI parser generation", id="generate-parser-button", disabled=True
                )
                yield Button("Delete Parser", id="delete-parser-button")
            yield Input(placeholder="Enter prompt (or let AI propose...)", id="prompt-input")
            yield Label("No active tab (interact to activate)", id="active-tab-url-label")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            table = self.query_one(DataTable)
            table.cursor_type = "row"
        except Exception as table_init_err:
            logger.error(f"Failed to initialize DataTable: {table_init_err}", exc_info=True)
        self.theme = DEFAULT_THEME

        # Set AI status on HomePanel
        try:
            home_panel = self.query_one(HomePanel)
            home_panel.ai_status = self._ai_status
        except Exception as ai_status_err:
            logger.error(
                f"Failed to set initial AI status on HomePanel: {ai_status_err}", exc_info=True
            )

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

        self._set_parser_button_enabled(False)  # Ensure disabled on start
        await self.action_check_chrome_status()

        # Disable AI buttons if no provider is configured
        if self._ai_status == "disabled":
            logger.info("AI is disabled, disabling AI-related buttons.")
            try:
                submit_button = self.query_one("#submit-button", Button)
                submit_button.disabled = True
                submit_button.tooltip = "AI disabled (set ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            except Exception as e:
                logger.error(f"Failed to disable submit button: {e}", exc_info=True)
            try:
                parser_button = self.query_one("#generate-parser-button", Button)
                parser_button.disabled = True
                parser_button.tooltip = "AI disabled (set ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            except Exception as e:
                logger.error(f"Failed to disable parser button: {e}", exc_info=True)
            try:
                delete_button = self.query_one("#delete-parser-button", Button)
                delete_button.disabled = True
                delete_button.tooltip = "AI disabled (set ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            except Exception as e:
                logger.error(f"Failed to disable delete button: {e}", exc_info=True)
            try:
                prompt_input = self.query_one("#prompt-input", Input)
                prompt_input.disabled = True
                prompt_input.placeholder = "AI Disabled (set ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            except Exception as e:
                logger.error(f"Failed to disable prompt input: {e}", exc_info=True)
        else:
            # Ensure parser button is disabled initially if AI *is* enabled
            self._set_parser_button_enabled(False)

    async def _handle_rehighlight(self, tab_ref: TabReference) -> None:
        # Pass the specific tab_ref to the trigger method
        await self.trigger_rehighlight(tab_ref)

    async def action_check_chrome_status(self) -> None:
        home_panel = self.query_one(HomePanel)
        home_panel.chrome_status = "checking"
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
        home_panel.chrome_status = new_status
        if new_status != "ready_to_connect":
            self._set_parser_button_enabled(False)
        if new_status == "ready_to_connect":
            self.app.call_later(self.action_connect_monitor)

    async def action_launch_chrome(self) -> None:
        logger.info("Action: Launching Chrome...")
        home_panel = self.query_one(HomePanel)
        home_panel.chrome_status = "checking"
        success = await chrome_launcher.launch_chrome()
        if not success:
            logger.error("Failed to launch Chrome via launcher.")
            home_panel.chrome_status = "error"
            return
        await asyncio.sleep(1.0)
        await self.action_check_chrome_status()

    async def action_restart_chrome(self) -> None:
        home_panel = self.query_one(HomePanel)
        home_panel.chrome_status = "checking"
        success = await chrome_launcher.restart_chrome_with_debug_port()
        if not success:
            logger.error("Failed to restart Chrome via launcher.")
            home_panel.chrome_status = "error"
            return
        await asyncio.sleep(1.0)
        await self.action_check_chrome_status()

    async def action_connect_monitor(self) -> None:
        home_panel = self.query_one(HomePanel)
        home_panel.chrome_status = "connecting"
        await asyncio.sleep(0.1)
        if not self._chrome_monitor:
            logger.error("Monitor not initialized, cannot connect.")
            home_panel.chrome_status = "error"
            return
        if not await chrome_launcher.is_chrome_debug_port_active():
            logger.error("Debug port became inactive before monitor could start.")
            await self.action_check_chrome_status()
            return
        try:
            # Ensure handler is instantiated before starting monitor
            if not self._monitor_handler:
                logger.error("MonitorEventHandler not initialized, cannot start monitor.")
                home_panel.chrome_status = "error"
                return

            success = await self._chrome_monitor.start_monitoring(
                on_polling_change_callback=self._monitor_handler.handle_polling_change,
                on_interaction_update_callback=self._monitor_handler.handle_interaction_update,
                on_content_fetched_callback=self._monitor_handler.handle_content_fetched,
            )
            if success:
                home_panel.chrome_status = "connected"
            else:
                logger.error("Failed to start Chrome Monitor.")
                home_panel.chrome_status = "error"
                self._set_parser_button_enabled(False)
        except Exception as e:
            logger.error(f"Error starting Chrome Monitor: {e}", exc_info=True)
            home_panel.chrome_status = "error"
            self._set_parser_button_enabled(False)

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
            submit_button = self.query_one("#submit-button", Button)
            if submit_button.label == "Stop AI selection":
                # --- Handle Stop Action --- #
                logger.info("User requested to stop AI selection.")
                if self._agent_worker and self._agent_worker.is_running:
                    logger.info("Cancelling active agent worker.")
                    self._agent_worker.cancel()
                    # Worker's CancelledError handler now manages UI state based on intermediate results.
                else:
                    logger.warning("Stop requested, but no agent worker found or running.")
                    # If no worker running, manually reset button just in case.
                    submit_button.label = "Start AI selection"
                    # Only enable if AI is not disabled
                    submit_button.disabled = self._ai_status == "disabled"
                    # Also ensure parser button is disabled if stop is pressed with no worker
                    self._set_parser_button_enabled(False)

            else:
                # --- Handle Start Action --- #
                # Prevent starting if AI is disabled
                if self._ai_status == "disabled":
                    logger.warning("AI features disabled. Cannot start selection.")
                    await self._update_ui_status("AI Disabled (No API key)", state="received_error")
                    return
                await self.on_input_submitted(Input.Submitted(input_widget, input_widget.value))
        elif button_id == "generate-parser-button":
            # Prevent starting if AI is disabled
            if self._ai_status == "disabled":
                logger.warning("AI features disabled. Cannot start parser generation.")
                await self._update_ui_status("AI Disabled (No API key)", state="received_error")
                return
            # Trigger parser generation workflow
            if self._codegen_worker and self._codegen_worker.is_running:
                logger.info("Cancelling previous codegen worker.")
                self._codegen_worker.cancel()

            # Run the codegen worker asynchronously
            self._codegen_worker = self.run_worker(
                self._run_parser_codegen_worker(),
                exclusive=True,
                group="parser_codegen",
            )
        elif button_id == "delete-parser-button":
            # --- Handle Delete Parser Action --- #
            logger.info("User requested to delete current source parser.")
            if self._monitor_handler and self._monitor_handler._current_parser_slug:
                slug_to_delete = self._monitor_handler._current_parser_slug
                deleted = self._monitor_handler._parser_registry.delete_source_parser(
                    slug_to_delete
                )
                if deleted:
                    logger.info(f"Successfully deleted parser with slug: {slug_to_delete}")
                    await self._update_ui_status("Parser deleted.", state="idle")
                    await self._clear_table_view()
                    # Hide the button via the handler method
                    self._monitor_handler._set_delete_button_visibility(False)
                    self.query_one("#delete-parser-button", Button).disabled = True
                    # Clear current parser info in handler
                    self._monitor_handler._current_parser_info = None
                    self._monitor_handler._current_parser_slug = None
                    # Explicitly clear the highlight for the deleted parser
                    if self._active_tab_ref:
                        await self._highlighter.clear_parser(self._active_tab_ref)
                else:
                    logger.error(f"Failed to delete parser with slug: {slug_to_delete}")
                    await self._update_ui_status("Failed to delete parser.", state="received_error")
        elif button_id == "drop-all-tables":
            # --- Handle Drop All Tables Action --- #
            logger.info("User requested to drop all DuckDB tables.")
            try:
                from selectron.cli.duckdb_utils import delete_all_tables

                # Run the deletion in a background thread to avoid blocking UI
                # (though it should be fast, good practice)
                async def _run_delete():
                    await asyncio.to_thread(delete_all_tables)
                    await self._update_ui_status("All tables dropped.", state="idle")

                self.run_worker(_run_delete(), exclusive=True, group="db_admin")
            except Exception as e:
                logger.error(f"Failed to run drop_all_tables: {e}", exc_info=True)
                await self._update_ui_status("Failed to drop tables", state="received_error")
        elif button_id == "open-duckdb":
            # --- Handle Open DuckDB UI Action --- #
            if self._duckdb_ui_conn:
                logger.info("DuckDB UI may already be running. Focusing existing browser window.")
                # Try to just open the URL again, browser might handle it
                webbrowser.open("http://localhost:4213")
                return

            try:
                from selectron.cli.duckdb_utils import get_db_path, launch_duckdb_ui

                db_path_str = str(get_db_path())
                conn = launch_duckdb_ui(db_path_str)
                if conn:
                    self._duckdb_ui_conn = conn  # Store the connection
                    logger.info("Stored DuckDB UI connection.")
                else:
                    logger.error("launch_duckdb_ui failed to return a connection.")
                    # Optionally show error to user
                    await self._update_ui_status("Failed to start DB UI", state="received_error")

            except Exception as e:
                logger.error(f"Failed to launch DuckDB UI: {e}", exc_info=True)
                await self._update_ui_status("Failed to start DB UI", state="received_error")
        else:
            logger.warning(f"Unhandled button press: {event.button.id}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt-input":
            selector_description = event.value.strip()
            if not selector_description:
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

            # Disable parser button when starting a new selection
            self._set_parser_button_enabled(False)

            # Update UI status immediately
            await self._update_ui_status("Preparing agent...", state="thinking")

            # Update button state: Change label and keep enabled for stopping
            try:
                submit_button = self.query_one("#submit-button", Button)
                submit_button.label = "Stop AI selection"
                submit_button.disabled = False  # Keep enabled to allow stopping
            except Exception as e:
                logger.error(f"Failed to update submit button state: {e}", exc_info=True)
                # Optionally handle the error, e.g., don't start the worker

            if self._agent_worker and self._agent_worker.is_running:
                logger.info("Cancelling previous agent worker.")
                self._agent_worker.cancel()

            self._agent_worker = self.run_worker(
                self._run_agent_worker(selector_description),
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

        # --- Stop DuckDB UI Server --- #
        if self._duckdb_ui_conn:
            logger.info("Stopping DuckDB UI server...")
            try:
                self._duckdb_ui_conn.execute("CALL stop_ui_server();")
                logger.info("DuckDB UI server stop command issued.")
            except Exception as stop_err:
                logger.error(f"Failed to stop DuckDB UI server: {stop_err}", exc_info=True)
            finally:
                try:
                    self._duckdb_ui_conn.close()
                    logger.info("Closed DuckDB UI connection.")
                except Exception as close_err:
                    logger.error(f"Failed to close DuckDB UI connection: {close_err}")
                self._duckdb_ui_conn = None

        self._highlighter.set_active(False)
        if self._active_tab_ref:
            try:
                # Schedule badge hide AND highlight clear on quit via call_later
                self.call_later(self._highlighter.hide_agent_status, self._active_tab_ref)
                self.call_later(self._highlighter.clear, self._active_tab_ref)
                self.call_later(self._highlighter.clear_parser, self._active_tab_ref)
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
            status_label = self.query_one("HomePanel #agent-status-label", Label)
            status_label.update(escape(message))
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

    async def _run_agent_worker(self, selector_description: str) -> None:
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

        try:
            submit_button = self.query_one("#submit-button", Button)
        except Exception as e:
            logger.error(f"Failed to query submit button: {e}", exc_info=True)
            submit_button = None

        async def status_callback(message: str, state: str, show_spinner: bool):
            await self._update_ui_status(message, state, show_spinner)

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
        agent: Optional[SelectorAgent] = None  # Store agent instance
        try:
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
                f"Running SelectorAgent for target '{selector_description}' on tab {tab_ref.id}"
            )
            proposal = await agent.run(selector_description)

            if proposal:
                await self._update_ui_status(
                    "Done",
                    state="final_success",
                    show_spinner=False,
                )
                self._last_proposed_selector = proposal.proposed_selector

                # Enable the parser button upon successful selection
                self._set_parser_button_enabled(True)

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
                # Reset button after successful completion
                if submit_button:
                    submit_button.label = "Start AI selection"
                    submit_button.disabled = False

        except SelectorAgentError as agent_err:
            # Agent already logged the specific error and updated status via callback
            # Set update_status=False because the agent's status_cb likely handled it
            await self._handle_agent_failure(
                f"SelectorAgent failed: {agent_err}", update_status=False
            )
        except asyncio.CancelledError:
            logger.info("Agent worker task was cancelled by user.")
            intermediate_selector: Optional[str] = None
            if agent:
                intermediate_selector = agent._best_selector_so_far

            if intermediate_selector and self._active_tab_ref:
                # --- Handle cancellation WITH a valid intermediate selector --- #
                logger.info(
                    f"Using intermediate selector found before cancellation: '{intermediate_selector}'"
                )
                self._last_proposed_selector = intermediate_selector

                # Use call_later for UI updates from worker
                self.call_later(
                    self._update_ui_status,
                    "Selection stopped; using intermediate result.",
                    "final_success",
                    False,
                )
                self.call_later(
                    self._highlighter.highlight, self._active_tab_ref, intermediate_selector, "lime"
                )
                self.call_later(self._set_parser_button_enabled, True)
                self.call_later(self._delayed_hide_status)  # Schedule status hide

                # Reset button immediately within call_later if possible, or schedule reset
                def _reset_button_on_cancel_success():
                    try:
                        button = self.query_one("#submit-button", Button)
                        button.label = "Start AI selection"
                        button.disabled = False
                    except Exception as e:
                        logger.error(
                            f"Failed to reset submit button on cancel success: {e}", exc_info=True
                        )

                self.call_later(_reset_button_on_cancel_success)

            else:
                # --- Handle cancellation WITHOUT a valid intermediate selector --- #
                logger.info("Agent cancelled, no intermediate selector found to use.")
                # Use call_later for UI updates from worker
                if self._active_tab_ref:
                    self.call_later(self._highlighter.clear, self._active_tab_ref)
                    self.call_later(self._highlighter.hide_agent_status, self._active_tab_ref)
                self.call_later(self._update_ui_status, "Selection cancelled.", "idle", False)
                self.call_later(self._set_parser_button_enabled, False)

                # Reset button
                def _reset_button_on_cancel_fail():
                    try:
                        button = self.query_one("#submit-button", Button)
                        button.label = "Start AI selection"
                        button.disabled = False
                    except Exception as e:
                        logger.error(
                            f"Failed to reset submit button on cancel fail: {e}", exc_info=True
                        )

                self.call_later(_reset_button_on_cancel_fail)

            # Do not re-raise cancellation error, as we've handled the state.
            # raise
        except Exception as e:
            # Catch unexpected errors *outside* the agent's known failure modes
            log_msg = f"Unexpected error in worker task for target '{selector_description}': {e}"
            # The handler will also update the status to a generic error message
            await self._handle_agent_failure(log_msg, update_status=True)
            # Button reset is handled by _handle_agent_failure
        finally:
            # Button reset logic is now handled within success/error/cancel paths
            pass
        # Note: Badge hiding is handled by success/error paths scheduling _delayed_hide_status
        # or within the CancelledError handler.

    async def trigger_rehighlight(self, tab_ref: Optional[TabReference] = None):
        # Check if there's an active tab and if the highlighter state indicates highlights are active
        # Use the provided tab_ref if available, otherwise fallback to the app's active tab
        target_ref = tab_ref or self._active_tab_ref

        if not target_ref:
            # logger.debug("Skipping rehighlight trigger: No target tab reference.")
            return

        # Call the highlighter's rehighlight method first to redraw overlays
        await self._highlighter.rehighlight(target_ref)

        # After rehighlighting, check if a parser needs to be re-applied
        if self._monitor_handler and target_ref.url:
            parser = None
            try:
                # Check if a parser exists for this URL by attempting to load it
                parser = self._monitor_handler._parser_registry.load_parser(target_ref.url)
                # No need to log if parser is found, only if not or error
            except FileNotFoundError:
                pass
                # This is expected if no parser is defined for the URL.
                # logger.debug(
                #     f"No parser found for url '{target_ref.url}' during rehighlight check."
                # )
                # parser remains None
            except Exception as e:
                logger.error(
                    f"Error loading parser for url '{target_ref.url}' during rehighlight check: {e}"
                )
                # parser remains None

            # If a valid parser dictionary was loaded, re-run the extraction logic
            if (
                parser
                and isinstance(parser, dict)
                and parser.get("selector")
                and parser.get("python")
            ):
                # The _apply_parser_extract method fetches live element HTML via CDP,
                # so it will use the elements visible/matching *after* the scroll/rehighlight.
                logger.debug(f"Re-applying parser extract for {target_ref.url} after rehighlight.")
                try:
                    # Ensure the handler still exists before calling its method
                    if self._monitor_handler:
                        await self._monitor_handler._apply_parser_extract(target_ref, parser)
                    else:
                        # Should technically not happen if we passed the initial check, but safety first.
                        logger.warning(
                            "Cannot re-apply parser extract: MonitorEventHandler became unavailable."
                        )
                except Exception as e:
                    logger.error(
                        f"Error during parser re-extraction triggered by rehighlight: {e}",
                        exc_info=True,
                    )
            # If parser is None or invalid, we simply don't re-run extraction.
        else:
            # Log why the parser re-apply check was skipped
            if not self._monitor_handler:
                logger.debug("Skipping parser re-apply check: MonitorEventHandler not available.")
            elif not (target_ref and target_ref.url):  # Combined check for clarity
                logger.debug("Skipping parser re-apply check: Target ref or its URL is missing.")

    async def _clear_table_view(self) -> None:
        try:
            table = self.query_one(DataTable)
            table.clear(columns=True)
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
            await self._highlighter.hide_agent_status(self._active_tab_ref)
        try:
            status_label = self.query_one("#agent-status-label", Label)
            status_label.update("Interact with a page in Chrome to get started")
        except Exception as e:
            logger.warning(f"Failed to reset status label after delay: {e}")

    async def _handle_agent_failure(self, log_message: str, update_status: bool = True) -> None:
        """Consolidated actions for when the selector agent fails."""
        logger.error(log_message, exc_info=True)  # Always include traceback for errors
        self._last_proposed_selector = None
        # Use call_later for UI updates from potentially non-main threads/tasks
        self.call_later(self._clear_table_view)
        self.call_later(self._delayed_hide_status)
        self.call_later(self._set_parser_button_enabled, False)
        if update_status:
            # Use call_later for status update as well
            error_msg = f"Agent Error: {log_message[:100]}..."  # Keep status concise
            self.call_later(self._update_ui_status, error_msg, "received_error", False)

        # Ensure button is reset on failure
        try:
            submit_button = self.query_one("#submit-button", Button)
            if submit_button.label == "Stop AI selection":
                submit_button.label = "Start AI selection"
                submit_button.disabled = False
        except Exception as e:
            logger.error(
                f"Failed to query/reset submit button in failure handler: {e}", exc_info=True
            )

    def _set_parser_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the 'Start AI parser generation' button, respecting AI status."""
        # Never enable if AI is globally disabled
        if self._ai_status == "disabled":
            enabled = False

        try:
            parser_button = self.query_one("#generate-parser-button", Button)
            parser_button.disabled = not enabled
            # Set tooltip based on why it's disabled
            if not enabled and self._ai_status == "disabled":
                parser_button.tooltip = "AI disabled (set ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            elif not enabled:
                parser_button.tooltip = "Requires a successful AI selection first"
            else:
                parser_button.tooltip = None  # Clear tooltip when enabled

        except Exception as e:
            logger.error(f"Failed to set parser button enabled state: {e}", exc_info=True)

    class _ChromeHighlighterAdapter(HighlighterProtocol):
        """Adapts ChromeHighlighter to the Highlighter protocol for a specific tab."""

        def __init__(self, chrome_highlighter: ChromeHighlighter, tab_ref: TabReference):
            self._highlighter = chrome_highlighter
            self._tab_ref = tab_ref

        async def highlight(self, selector: str, color: str) -> bool:
            return await self._highlighter.highlight(self._tab_ref, selector, color)

        async def clear(self) -> None:
            await self._highlighter.clear(self._tab_ref)

        async def show_agent_status(self, text: str, state: str, show_spinner: bool) -> None:
            await self._highlighter.show_agent_status(self._tab_ref, text, state, show_spinner)

        async def hide_agent_status(self) -> None:
            await self._highlighter.hide_agent_status(self._tab_ref)

    async def _run_parser_codegen_worker(self) -> None:
        """Worker task to run CodegenAgent for parser generation."""

        # Preconditions: we need an active tab, a selector, and HTML samples.
        if not self._active_tab_ref:
            logger.warning("Cannot run CodegenAgent: no active tab.")
            await self._update_ui_status(
                "Parser generation error: no active tab",
                state="received_error",
                show_spinner=False,
            )
            return

        if not self._last_proposed_selector:
            logger.warning("Cannot run CodegenAgent: no selector proposal available.")
            await self._update_ui_status(
                "Parser generation error: no selector",
                state="received_error",
                show_spinner=False,
            )
            return

        selector = self._last_proposed_selector
        tab_ref = self._active_tab_ref

        # Retrieve sample HTML snippets for the selector
        try:
            html_samples = await self._highlighter.get_elements_html(
                tab_ref, selector, max_elements=3
            )
        except Exception as e:
            logger.error(f"Failed to retrieve HTML samples for codegen: {e}", exc_info=True)
            await self._update_ui_status(
                "Parser generation error: failed to fetch elements",
                state="received_error",
                show_spinner=False,
            )
            return

        if not html_samples:
            logger.warning("CodegenAgent: selector matched no elements – aborting.")
            await self._update_ui_status(
                "Parser generation error: no elements matched",
                state="received_error",
                show_spinner=False,
            )
            return

        # Grab the selector description from prompt input if available
        try:
            prompt_input = self.query_one("#prompt-input", Input)
            selector_description = prompt_input.value.strip()
        except Exception as e:
            logger.error(f"Failed to retrieve prompt input for codegen: {e}", exc_info=True)
            selector_description = ""

        # Disable parser button while running
        try:
            parser_button = self.query_one("#generate-parser-button", Button)
            parser_button.label = "Running AI..."
            parser_button.disabled = True
        except Exception as e:
            logger.error(f"Failed to update parser button state at start: {e}", exc_info=True)
            parser_button = None  # Ensure it's None if query fails

        # Update UI status
        await self._update_ui_status(
            "Generating parser code… (this may take a minute)", state="thinking", show_spinner=True
        )

        # Run CodegenAgent
        try:
            codegen_agent = CodegenAgent(
                html_samples=html_samples,
                model_cfg=self._model_config,
                save_results=True,
                base_url=tab_ref.url or "",
                input_selector=selector,
                input_selector_description=selector_description,
                status_cb=self._update_ui_status,
            )

            logger.info(f"Starting CodegenAgent for url '{tab_ref.url}' with selector '{selector}'")

            # Capture agent messages
            with capture_run_messages() as messages:
                generated_code, outputs = await codegen_agent.run()

            _ = (generated_code, outputs)  # silence unused variable lints

            logger.info("CodegenAgent finished successfully. Triggering parser reload.")

            if self._monitor_handler:
                try:
                    self._monitor_handler._parser_registry.rescan_parsers()
                    # Schedule the check/apply for the current tab
                    if self._active_tab_ref:
                        active_ref_capture = self._active_tab_ref  # Capture for closure
                        self.call_later(
                            self._monitor_handler._maybe_apply_parser_highlight, active_ref_capture
                        )
                except Exception as reload_err:
                    logger.error(f"Error rescanning/applying parser: {reload_err}", exc_info=True)

            await self._update_ui_status(
                "Parser generated and saved.",
                state="final_success",
                show_spinner=False,
            )

            # Optionally, re-enable parser button if we want to regenerate again
            self._set_parser_button_enabled(True)

        except UnexpectedModelBehavior as e:
            logger.error(f"CodegenAgent failed with UnexpectedModelBehavior: {e}", exc_info=True)
            logger.error("Captured Agent Messages (on failure):")
            # Log each message for better readability in the log file
            for msg in messages:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
                logger.error(f"  - {role}: {str(content)[:500]}...")  # Log truncated message
            await self._update_ui_status(
                f"Parser generation failed: {e}",  # Show error to user
                state="received_error",
                show_spinner=False,
            )
            # Keep button disabled on error

        except Exception as e:
            logger.error(f"CodegenAgent failed: {e}", exc_info=True)
            await self._update_ui_status(
                "Parser generation failed",
                state="received_error",
                show_spinner=False,
            )
            # Keep button disabled on error

        finally:
            # Ensure spinner/badge gets hidden eventually
            self.call_later(self._delayed_hide_status)
            # Finished: Reset button state
            if parser_button:
                parser_button.label = "Start AI parser generation"
                # Enable state is handled within try/except blocks above
                # For success: self._set_parser_button_enabled(True)
                # For error: remains disabled (no explicit enable)
                # If we want to ALWAYS re-enable, we'd do it here.
                # Let's stick to re-enabling only on success for now.
                # If AI is enabled, enable the button, otherwise _set_parser_button_enabled handles it
                if self._ai_status != "disabled":
                    self._set_parser_button_enabled(True)

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from PIL import Image

from selectron.ai.propose_selection import propose_selection
from selectron.ai.types import AutoProposal
from selectron.chrome.chrome_highlighter import ChromeHighlighter
from selectron.chrome.chrome_monitor import TabChangeEvent
from selectron.chrome.types import TabReference
from selectron.parse.parser_registry import ParserRegistry
from selectron.util.logger import get_logger

if TYPE_CHECKING:
    from textual.widgets import DataTable, Input, Label

    from selectron.cli.app import SelectronApp  # Use specific type if possible

logger = get_logger(__name__)


class MonitorEventHandler:
    """Handles callbacks from the ChromeMonitor."""

    def __init__(
        self,
        app: SelectronApp,  # Keep strong typing
        highlighter: ChromeHighlighter,
        url_label: Label,
        data_table: DataTable,
        prompt_input: Input,
    ):
        self._app = app
        self._highlighter = highlighter
        # Store references to UI elements needed
        self._url_label = url_label
        self._data_table = data_table
        self._prompt_input = prompt_input

        # --- Parser related --- #
        self._parser_registry = ParserRegistry()
        # Track last URL for which parser highlight was applied per tab
        self._parser_highlighted_for_tab: dict[str, str] = {}

    async def handle_polling_change(self, event: TabChangeEvent) -> None:
        """Handles tab navigation/changes detected by polling."""
        # Logic moved from SelectronApp._handle_polling_change
        navigated_tabs_info = event.navigated_tabs
        if navigated_tabs_info:
            navigated_ids = {new_tab.id for new_tab, _ in navigated_tabs_info}
            logger.debug(f"Polling detected navigation for tabs: {navigated_ids}. Resetting flag.")
            # Access app state via self._app
            self._app._propose_selection_done_for_tab = None
            # NOTE: Cannot reliably clear highlights/badges here as ws_url may be missing

            # clear any persistent parser highlights for navigated tabs
            for _, old_ref in navigated_tabs_info:
                try:
                    await self._highlighter.clear_parser(old_ref)
                except Exception as e:
                    logger.debug(
                        f"Failed to clear parser highlight on navigation for tab {old_ref.id}: {e}"
                    )
                # remove tracking
                self._parser_highlighted_for_tab.pop(old_ref.id, None)

    async def handle_interaction_update(self, tab_ref: TabReference) -> None:
        """Handles updates triggered by user interaction in a tab."""
        # Logic moved from SelectronApp._handle_interaction_update
        self._app._active_tab_ref = tab_ref

        try:
            # Use the stored Label reference
            if tab_ref and tab_ref.url:
                self._url_label.update(tab_ref.url)
        except Exception as label_err:
            logger.error(f"Failed to update URL label on interaction: {label_err}")

        # Call app's method to clear table
        await self._app._clear_table_view()

        # Reset proposal flag logic
        current_active_id_for_propose_select = self._app._propose_selection_done_for_tab
        if tab_ref and tab_ref.id != current_active_id_for_propose_select:
            logger.debug(
                f"Interaction detected on tab {tab_ref.id} (different from last proposal tab {current_active_id_for_propose_select}). Priming proposal."
            )
            self._app._propose_selection_done_for_tab = (
                None  # Allow proposal for this tab upon next content fetch
            )

        # if user interacted but url unchanged, ensure parser highlight present
        if tab_ref and tab_ref.url:
            await self._maybe_apply_parser_highlight(tab_ref)

    async def handle_content_fetched(
        self,
        tab_ref: TabReference,
        screenshot: Optional[Image.Image],
        scroll_y: Optional[int],
        dom_string: Optional[str],
    ) -> None:
        """Handles updates after tab content (HTML, screenshot, DOM) is fetched."""
        # Logic moved from SelectronApp._handle_content_fetched
        html_content = tab_ref.html  # HTML is now part of the TabReference passed

        if not html_content:
            logger.warning(
                f"Monitor Content Fetched (Tab {tab_ref.id}): No HTML content in TabReference."
            )
            await self._app._clear_table_view()
            return

        # Always update the active ref and DOM string first (on the app)
        self._app._active_tab_ref = tab_ref
        self._app._active_tab_dom_string = dom_string

        # Update UI Label using the latest fetched info (using stored ref)
        try:
            if tab_ref and tab_ref.url:
                self._url_label.update(tab_ref.url)
            else:
                self._url_label.update("No active tab URL")
        except Exception as label_err:
            logger.error(f"Failed to update URL label on content fetch: {label_err}")

        # NOTE: table clearing and population is now handled by _apply_parser_extract or _clear_table_view

        # Check if the main agent worker is running (on the app)
        if self._app._agent_worker and self._app._agent_worker.is_running:
            logger.debug("Skipping proposal: Selector agent is currently running.")
            pass  # Let the logic proceed to potentially update the flag if needed, but don't start worker
        else:
            # Only proceed with proposal if agent is NOT running
            if (
                screenshot
                and self._app._active_tab_ref
                and self._app._propose_selection_done_for_tab != self._app._active_tab_ref.id
            ):
                # Use app's _update_ui_status helper instead of direct highlighter call
                await self._app._update_ui_status(
                    "Proposing selection...", state="thinking", show_spinner=True
                )

                if (
                    self._app._propose_selection_worker
                    and self._app._propose_selection_worker.is_running
                ):
                    logger.debug("Cancelling previous propose worker.")
                    self._app._propose_selection_worker.cancel()

                async def _do_propose_selection():
                    try:
                        # Use app's model config
                        proposal = await propose_selection(screenshot, self._app._model_config)
                        if self._app._active_tab_ref:
                            self._app._propose_selection_done_for_tab = self._app._active_tab_ref.id

                        if isinstance(proposal, AutoProposal):
                            desc = proposal.proposed_description

                            def _update_input():
                                try:
                                    # Use stored Input ref
                                    self._prompt_input.value = desc
                                    if desc and self._app._active_tab_ref:
                                        # Use app's call_later
                                        self._app.call_later(
                                            self._app._update_ui_status,
                                            desc,
                                            "idle",
                                            False,  # No spinner for idle state
                                        )
                                except Exception as e:
                                    logger.error(f"Error updating input from proposal: {e}")

                            # Use app's call_later
                            self._app.call_later(_update_input)

                            if self._app._active_tab_ref:
                                self._app.call_later(
                                    self._app._update_ui_status,
                                    desc,
                                    "idle",
                                    False,  # No spinner for idle state
                                )

                    except Exception as e:
                        logger.error(f"Error during proposal: {e}", exc_info=True)
                        if self._app._active_tab_ref:
                            self._app.call_later(
                                lambda: self._highlighter.hide_agent_status(
                                    self._app._active_tab_ref
                                )
                            )

                # Use app's run_worker
                self._app._propose_selection_worker = self._app.run_worker(
                    _do_propose_selection(), exclusive=True, group="propose_selection"
                )
            elif (
                screenshot
                and self._app._active_tab_ref
                and self._app._propose_selection_done_for_tab == self._app._active_tab_ref.id
            ):
                pass
            elif not screenshot:
                logger.debug("Skipping proposal: No screenshot available.")

        # --- Parser highlight logic (run after HTML present) --- #
        await self._maybe_apply_parser_highlight(tab_ref)

    # --- Internal helper for parser highlight --- #
    async def _maybe_apply_parser_highlight(self, tab_ref: TabReference) -> None:
        """Load parser for current url and apply highlight if appropriate."""
        if not tab_ref.url or not tab_ref.id:
            return

        last_url = self._parser_highlighted_for_tab.get(tab_ref.id)
        if last_url == tab_ref.url:
            # already highlighted for this url
            return

        # load parser
        try:
            parser = self._parser_registry.load_parser(tab_ref.url)
        except Exception as e:
            logger.error(f"Error loading parser for url '{tab_ref.url}': {e}")
            parser = None

        # clear previous highlight first (if any)
        await self._highlighter.clear_parser(tab_ref)

        if (
            parser
            and isinstance(parser, dict)
            and (selector := parser.get("selector"))
            and isinstance(selector, str)
        ):
            try:
                success = await self._highlighter.highlight_parser(tab_ref, selector, color="cyan")
                if success:
                    self._parser_highlighted_for_tab[tab_ref.id] = tab_ref.url

                    # After successful highlight, run parser extraction and update table
                    if parser and isinstance(parser.get("python"), str):
                        await self._apply_parser_extract(tab_ref, parser)

                else:
                    self._parser_highlighted_for_tab.pop(tab_ref.id, None)
            except Exception as e:
                logger.debug(f"Failed to apply parser highlight for tab {tab_ref.id}: {e}")
                self._parser_highlighted_for_tab.pop(tab_ref.id, None)
        else:
            # no parser – ensure mapping cleared
            self._parser_highlighted_for_tab.pop(tab_ref.id, None)

    # --- Run parser code on selected elements and update data table --- #
    async def _apply_parser_extract(self, tab_ref: TabReference, parser: dict) -> None:
        """Execute parser python code against each selected element's HTML (fetched live) and display results as columns."""

        import json
        import reprlib

        from bs4 import BeautifulSoup

        selector = parser.get("selector")
        python_code = parser.get("python")

        if not selector or not python_code or not isinstance(selector, str):
            logger.debug("_apply_parser_extract: missing selector or python code.")
            return

        # Prepare sandbox
        sandbox: dict[str, Any] = {"BeautifulSoup": BeautifulSoup, "json": json}
        try:
            exec(python_code, sandbox)
        except Exception as e:
            logger.error(f"Parser execution error: {e}", exc_info=True)
            return

        parse_fn = sandbox.get("parse_element")
        if not callable(parse_fn):
            logger.error("Parser does not define a callable 'parse_element' function.")
            return

        # Get element HTML directly from the browser
        try:
            element_htmls = await self._highlighter.get_elements_html(
                tab_ref, selector, max_elements=100
            )
        except Exception as e:
            logger.error(f"Error getting element HTML via CDP: {e}", exc_info=True)
            await self._app._clear_table_view()  # Clear table on error
            return

        if not element_htmls:
            logger.debug(f"Parser selector '{selector}' matched no elements in live browser.")
            await self._app._clear_table_view()  # clear table if no results
            return

        # Execute parser on each element and collect results
        results_data: list[tuple[str, dict | None]] = []  # (html_snippet, parsed_dict)
        for idx, outer_html in enumerate(element_htmls):
            html_display = (
                outer_html[:500] + "..." if len(outer_html) > 500 else outer_html
            )  # shorten html display
            parsed_dict = None
            try:
                parsed_dict = parse_fn(outer_html)
            except Exception as e:
                logger.error(f"Error running parse_element for element {idx}: {e}", exc_info=True)
            results_data.append((html_display, parsed_dict))

        # Determine columns from the first valid result dictionary
        column_keys: list[str] = []
        for _, parsed_dict in results_data:
            if isinstance(parsed_dict, dict):
                column_keys = list(parsed_dict.keys())
                break

        # Update data table
        try:
            table = self._data_table
            table.clear(columns=True)

            # Add columns
            table.add_column("Raw HTML", key="_raw_html_")
            for key in column_keys:
                table.add_column(key.replace("_", " ").title(), key=key)

            # Add rows
            repr_short = reprlib.Repr()
            repr_short.maxstring = 50
            repr_short.maxother = 50

            for i, (html_snippet, parsed_dict) in enumerate(results_data):
                row_data = [html_snippet]
                if isinstance(parsed_dict, dict):
                    for key in column_keys:
                        # represent value concisely
                        value = parsed_dict.get(key)
                        row_data.append(repr_short.repr(value))
                else:
                    # Add placeholders if parsing failed or returned non-dict
                    row_data.extend(["-"] * len(column_keys))

                table.add_row(*row_data, key=f"parsed_{tab_ref.id}_{i}")

        except Exception as e:
            logger.error(f"Failed to update data table with parser results: {e}", exc_info=True)

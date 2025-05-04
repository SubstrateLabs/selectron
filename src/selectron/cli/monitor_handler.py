from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from PIL import Image
from textual.widgets import Button, DataTable, Input, Label

from selectron.ai.propose_selection import propose_selection
from selectron.ai.types import AutoProposal
from selectron.chrome.chrome_highlighter import ChromeHighlighter
from selectron.chrome.chrome_monitor import TabChangeEvent
from selectron.chrome.types import TabReference
from selectron.parse.parser_registry import ParserRegistry
from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

if TYPE_CHECKING:
    from textual.widgets import Button, DataTable, Input, Label

    from selectron.cli.app import SelectronApp  # Use specific type if possible

logger = get_logger(__name__)


class MonitorEventHandler:
    """Handles callbacks from the ChromeMonitor."""

    def __init__(
        self,
        *,  # Force keyword args
        app: "SelectronApp",  # Forward reference for typing
        highlighter: ChromeHighlighter,
        url_label: Label,
        data_table: DataTable,
        prompt_input: Input,
    ):
        self.app = app
        self._highlighter = highlighter
        # Store references to UI elements needed
        self._url_label = url_label
        self._data_table = data_table
        self._prompt_input = prompt_input

        # --- Parser related --- #
        self._parser_registry = ParserRegistry()
        # Track last URL for which parser highlight was applied per tab
        self._parser_highlighted_for_tab: dict[str, str] = {}
        self._last_polling_state: Dict[int, str] = {}
        self._last_interaction_update_time: float = 0
        self._current_parser_info: Optional[Tuple[Dict[str, Any], str, Path]] = (
            None  # (parser_dict, origin, path)
        )
        self._current_parser_slug: Optional[str] = None

    async def handle_polling_change(self, event: TabChangeEvent) -> None:
        """Handles tab navigation/changes detected by polling."""
        # Logic moved from SelectronApp._handle_polling_change
        navigated_tabs_info = event.navigated_tabs
        if navigated_tabs_info:
            navigated_ids = {new_tab.id for new_tab, _ in navigated_tabs_info}
            logger.debug(f"Polling detected navigation for tabs: {navigated_ids}. Resetting flag.")
            # Access app state via self.app
            self.app._propose_selection_done_for_tab = None
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
        self.app._active_tab_ref = tab_ref

        try:
            # Use the stored Label reference
            if tab_ref and tab_ref.url:
                self._url_label.update(tab_ref.url)
        except Exception as label_err:
            logger.error(f"Failed to update URL label on interaction: {label_err}")

        # Call app's method to clear table
        await self.app._clear_table_view()

        # Reset proposal flag logic
        current_active_id_for_propose_select = self.app._propose_selection_done_for_tab
        if tab_ref and tab_ref.id != current_active_id_for_propose_select:
            self.app._propose_selection_done_for_tab = (
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
            await self.app._clear_table_view()
            return

        # Always update the active ref and DOM string first (on the app)
        self.app._active_tab_ref = tab_ref
        self.app._active_tab_dom_string = dom_string

        # Update UI Label using the latest fetched info (using stored ref)
        try:
            if tab_ref and tab_ref.url:
                self._url_label.update(tab_ref.url)
            else:
                self._url_label.update("No active tab URL")
        except Exception as label_err:
            logger.error(f"Failed to update URL label on content fetch: {label_err}")

        # NOTE: table clearing and population is now handled by _apply_parser_extract or _clear_table_view

        # --- Check for existing parser before proposing ---
        parser_found_and_handled = False
        parser = None
        if tab_ref and tab_ref.url:
            try:
                parser = self._parser_registry.load_parser(tab_ref.url)
                if parser and isinstance(parser, dict):
                    selector_description = parser.get("selector_description")
                    if selector_description and isinstance(selector_description, str):
                        logger.info(
                            f"Parser found for {tab_ref.url}, using description: '{selector_description}'"
                        )
                        # ---> Cancel any pending debounce timer BEFORE setting value <--- #
                        if self.app._input_debounce_timer:
                            self.app._input_debounce_timer.stop()
                            self.app._input_debounce_timer = None  # Clear the reference
                        self._prompt_input.value = selector_description  # Now set the value
                        # Update status immediately and mark proposal as 'done' for this tab
                        self.app._propose_selection_done_for_tab = tab_ref.id
                        parser_found_and_handled = True
                    else:
                        logger.debug(
                            f"Parser found for {tab_ref.url} but has no 'selector_description'."
                        )
                # No need to log FileNotFoundError, it's expected
            except FileNotFoundError:
                pass  # No parser exists, proceed to proposal logic if applicable
            except Exception as e:
                logger.error(f"Error checking for parser for url '{tab_ref.url}': {e}")

        # Check if the main agent worker is running (on the app)
        if self.app._agent_worker and self.app._agent_worker.is_running:
            logger.debug("Skipping proposal: Selector agent is currently running.")
            pass  # Let the logic proceed to potentially update the flag if needed, but don't start worker
        elif not parser_found_and_handled:  # Only propose if parser wasn't found/handled
            # Only proceed with proposal if agent is NOT running
            if (
                screenshot
                and self.app._active_tab_ref
                and self.app._propose_selection_done_for_tab != self.app._active_tab_ref.id
            ):
                # Use app's _update_ui_status helper instead of direct highlighter call
                await self.app._update_ui_status(
                    "Proposing selection...", state="thinking", show_spinner=True
                )

                if (
                    self.app._propose_selection_worker
                    and self.app._propose_selection_worker.is_running
                ):
                    logger.debug("Cancelling previous propose worker.")
                    self.app._propose_selection_worker.cancel()

                async def _do_propose_selection():
                    try:
                        # Use app's model config
                        proposal = await propose_selection(screenshot, self.app._model_config)
                        if self.app._active_tab_ref:
                            self.app._propose_selection_done_for_tab = self.app._active_tab_ref.id

                        if isinstance(proposal, AutoProposal):
                            desc = proposal.proposed_description
                            try:
                                self._prompt_input.value = desc
                                await self.app._update_ui_status(desc, "idle", False)
                            except Exception as ui_update_err:
                                logger.error(
                                    f"Error during DIRECT UI update from proposal worker: {ui_update_err}",
                                    exc_info=True,
                                )

                        else:
                            logger.warning(
                                f"propose_selection returned unexpected type: {type(proposal)}"
                            )
                            # Optionally hide status or show generic message if proposal is not AutoProposal
                            if self.app._active_tab_ref:
                                self.app.call_later(
                                    lambda: self._highlighter.hide_agent_status(
                                        self.app._active_tab_ref
                                    )
                                )

                    except Exception as e:
                        logger.error(f"Error during proposal worker: {e}", exc_info=True)
                        # Explicitly log if this generic exception handler is hit
                        logger.critical(
                            "*** Generic exception handler in _do_propose_selection was triggered ***"
                        )
                        if self.app._active_tab_ref:
                            # Attempt to hide status even on failure, schedule it via call_later
                            tab_ref_capture = self.app._active_tab_ref  # Capture ref
                            self.app.call_later(
                                lambda: asyncio.create_task(self._try_hide_status(tab_ref_capture))
                            )

                # Use app's run_worker
                self.app._propose_selection_worker = self.app.run_worker(
                    _do_propose_selection(), exclusive=True, group="propose_selection"
                )
            elif (
                screenshot
                and self.app._active_tab_ref
                and self.app._propose_selection_done_for_tab == self.app._active_tab_ref.id
            ):
                pass
            elif not screenshot:
                logger.debug("Skipping proposal: No screenshot available.")

        # --- Parser highlight logic (run after HTML present) --- #
        await self._maybe_apply_parser_highlight(tab_ref)

    async def _try_hide_status(self, tab_ref: TabReference) -> None:
        """Attempt to hide status badge, catching errors."""
        try:
            logger.debug(
                f"Attempting to hide status badge for tab {tab_ref.id} after proposal failure."
            )
            await self._highlighter.hide_agent_status(tab_ref)
        except Exception as hide_err:
            logger.error(f"Error trying to hide agent status: {hide_err}", exc_info=True)

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

        self._set_delete_button_visibility(False)  # Hide by default
        self._current_parser_info = None
        self._current_parser_slug = None

        # If parser found, apply highlight and extract
        if parser:
            parser_dict, origin, _file_path = parser
            self._current_parser_info = parser
            self._current_parser_slug = slugify_url(tab_ref.url)

            logger.info(f"Applying parser (origin: {origin}) for URL {tab_ref.url}")
            selector = parser_dict.get("selector")
            if selector:
                # Highlight elements matched by the parser's selector
                await self._highlighter.highlight_parser(tab_ref, selector)
                self._parser_highlighted_for_tab[tab_ref.id] = tab_ref.url  # Mark highlight done

                # Extract data using the parser and update the table
                await self._apply_parser_extract(tab_ref, parser_dict)

                # Show delete button ONLY if the parser origin is 'source'
                self._set_delete_button_visibility(origin == "source")

            else:
                logger.warning(f"Loaded parser for {tab_ref.url} is missing 'selector' key.")
                self._set_delete_button_visibility(False)
        else:
            logger.debug(f"No parser found for {tab_ref.url}, clearing parser highlights.")
            await self._highlighter.clear_parser(tab_ref)
            self._set_delete_button_visibility(False)

    # --- Run parser code on selected elements and update data table --- #
    async def _apply_parser_extract(
        self, tab_ref: TabReference, parser_dict: Dict[str, Any]
    ) -> None:
        """Execute parser python code against each selected element's HTML (fetched live) and display results as columns."""

        import json
        import reprlib

        from bs4 import BeautifulSoup

        selector = parser_dict.get("selector")
        python_code = parser_dict.get("python")

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
            await self.app._clear_table_view()  # Clear table on error
            return

        if not element_htmls:
            logger.debug(f"Parser selector '{selector}' matched no elements in live browser.")
            await self.app._clear_table_view()  # clear table if no results
            return

        # Execute parser on each element and collect results
        results_data: list[dict[str, Any] | None] = []  # Explicit key/value type
        for idx, outer_html in enumerate(element_htmls):
            parsed_dict: dict[str, Any] | None = None  # Initialize with explicit type
            try:
                result = parse_fn(outer_html)
                # Validate the result is a dictionary before assigning
                if isinstance(result, dict):
                    parsed_dict = result
                else:
                    logger.error(
                        f"parse_element for element {idx} returned non-dict type: {type(result).__name__}"
                    )
                    # Keep parsed_dict as None
            except Exception as e:
                logger.error(f"Error running parse_element for element {idx}: {e}", exc_info=True)
                # parsed_dict remains None

            results_data.append(parsed_dict)

        # Determine columns from the first valid result dictionary
        column_keys: list[str] = []
        for parsed_dict in results_data:  # Iterate only over dicts
            if isinstance(parsed_dict, dict):
                column_keys = list(parsed_dict.keys())
                break

        # Update data table
        try:
            table = self._data_table
            table.clear(columns=True)

            # Add columns
            for key in column_keys:
                table.add_column(key, key=key)

            # Add rows
            repr_short = reprlib.Repr()
            repr_short.maxstring = 50
            repr_short.maxother = 50

            for i, parsed_dict in enumerate(results_data):  # Unpack only the dict
                row_data = []  # Changed initialization from [html_snippet]
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

    def _set_delete_button_visibility(self, visible: bool) -> None:
        """Sets the visibility/disabled state of the delete parser button in the app."""
        try:
            # Access the button through the app reference
            delete_button = self.app.query_one("#delete-parser-button", Button)
            # Hide button by setting display=False, or disable it
            # Using display: none is usually better for layout stability
            delete_button.display = visible
            # Alternatively, disable: delete_button.disabled = not visible
        except Exception as e:
            # Log if the button isn't found (might happen during setup/teardown)
            logger.debug(f"Could not find or update delete-parser-button: {e}")

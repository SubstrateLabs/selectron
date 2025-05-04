import asyncio
from typing import Awaitable, Callable, Optional

import openai
import websockets
from PIL import Image

from selectron.chrome.cdp_executor import CdpBrowserExecutor
from selectron.chrome.chrome_cdp import (
    ChromeTab,
    capture_tab_screenshot,
    get_final_url_and_title,
    get_html_via_ws,
    wait_for_page_load,
)
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent
from selectron.chrome.connect import ensure_chrome_connection
from selectron.chrome.types import TabReference
from selectron.dom.dom_attributes import DOM_STRING_INCLUDE_ATTRIBUTES
from selectron.dom.dom_service import DomService
from selectron.util.logger import get_logger

logger = get_logger(__name__)


# Define callback types for clarity
ActiveTabCallback = Callable[[Optional[TabReference], Optional[str]], Awaitable[None]]
PageContentCallback = Callable[[str], Awaitable[None]]


class ChromeTabManager:
    """Manages Chrome monitoring, tab events, data fetching, and proposal generation."""

    monitor: Optional[ChromeMonitor] = None
    shutdown_event: asyncio.Event
    _active_tab_ref: Optional[TabReference] = None
    _openai_client: Optional[openai.AsyncOpenAI] = None

    # Callbacks provided by the UI layer
    _on_active_tab_updated: ActiveTabCallback
    _on_page_content_ready: PageContentCallback

    def __init__(
        self,
        openai_client: Optional[openai.AsyncOpenAI],
        on_active_tab_updated: ActiveTabCallback,
        on_page_content_ready: PageContentCallback,
    ):
        self.shutdown_event = asyncio.Event()
        self._openai_client = openai_client
        self._on_active_tab_updated = on_active_tab_updated
        self._on_page_content_ready = on_page_content_ready
        logger.info("ChromeTabManager initialized.")

    async def start_monitoring_loop(self) -> bool:
        """Initializes and starts the Chrome monitor loop."""
        logger.info("Initializing Chrome monitor within Tab Manager...")
        # Ensure Chrome connection first
        if not await ensure_chrome_connection():
            logger.error("Tab Manager: Failed to establish Chrome connection.")
            return False

        # Instantiate the monitor
        self.monitor = ChromeMonitor(
            # Pass a no-op async lambda for the required callback
            rehighlight_callback=lambda _: asyncio.sleep(0),  # Accept one arg (_)
            check_interval=1.5,  # Or make configurable
        )

        try:
            logger.info("Tab Manager: Starting ChromeMonitor...")
            monitor_started = await self.monitor.start_monitoring(
                on_polling_change_callback=self._handle_polling_change,
                on_interaction_update_callback=self._handle_interaction_update,
                on_content_fetched_callback=self._handle_content_fetched,
            )
            if not monitor_started:
                logger.error("Tab Manager: Failed to start ChromeMonitor.")
                self.monitor = None  # Clear monitor instance on failure
                return False

            logger.info("Tab Manager: ChromeMonitor started successfully.")
            return True
        except Exception as e:
            logger.exception(f"Tab Manager: Error during monitor startup: {e}")
            self.monitor = None  # Ensure monitor is cleared on exception
            return False

    async def run_monitoring_task(self) -> None:
        """The main task function to be run as a worker."""
        # Start the monitor loop first
        monitor_started = await self.start_monitoring_loop()
        if not monitor_started:
            logger.error("Tab Manager: Monitoring task exiting because monitor failed to start.")
            # Signal shutdown immediately if start fails, so the worker exits
            self.shutdown_event.set()
            return  # Exit the task

        # If monitor started successfully, wait for the shutdown signal
        try:
            logger.info("Tab Manager: Monitoring loop started. Waiting for shutdown signal...")
            await self.shutdown_event.wait()
            logger.info("Tab Manager: Shutdown signal received.")
        except asyncio.CancelledError:
            logger.info("Tab Manager: Monitoring task cancelled.")
            # Ensure shutdown event is set if cancelled externally
            if not self.shutdown_event.is_set():
                self.shutdown_event.set()
        except Exception as e:
            logger.exception(f"Tab Manager: Unexpected error in monitoring wait loop: {e}")
            # Ensure shutdown event is set on error
            if not self.shutdown_event.is_set():
                self.shutdown_event.set()
        finally:
            # Cleanup happens in stop_monitoring, which should be called externally
            # or triggered by setting the shutdown_event.
            logger.info("Tab Manager: Monitoring task finishing.")
            # No need to call stop_monitoring here, as it's called from action_quit
            # or potentially another external trigger.

    async def stop_monitoring(self) -> None:
        """Signals the monitoring loop to stop and cleans up."""
        logger.info("Stopping Chrome monitoring...")
        self.shutdown_event.set()

        # Stop the underlying monitor
        if self.monitor:
            await self.monitor.stop_monitoring()
        logger.info("ChromeTabManager stopped.")

    async def _handle_polling_change(self, event: TabChangeEvent):
        """Callback function for tab changes detected ONLY by polling."""
        tasks = []
        # Determine if the *internally tracked* active tab was affected
        active_tab_closed_or_navigated = False
        if self._active_tab_ref:
            # Check closed tabs
            if any(closed_ref.id == self._active_tab_ref.id for closed_ref in event.closed_tabs):
                active_tab_closed_or_navigated = True
            # Check navigated tabs (old reference)
            if not active_tab_closed_or_navigated and any(
                old_ref.id == self._active_tab_ref.id for _, old_ref in event.navigated_tabs
            ):
                active_tab_closed_or_navigated = True

        # If the active tab was affected, clear the internal reference.
        # The UI layer will handle UI changes (like clearing highlights) via the callback.
        if active_tab_closed_or_navigated:
            logger.info("Polling: Active tab closed or navigated, clearing internal reference.")
            # Clear internal state first
            active_tab_id = self._active_tab_ref.id if self._active_tab_ref else "unknown"
            self._active_tab_ref = None
            # Notify UI layer that the active tab is now None
            try:
                await self._on_active_tab_updated(None, None)
            except Exception as cb_err:
                logger.error(
                    f"Error invoking `_on_active_tab_updated(None)` for closed/navigated tab {active_tab_id}: {cb_err}"
                )

        # Process new and navigated tabs by calling internal method
        for new_tab in event.new_tabs:
            if new_tab.webSocketDebuggerUrl:
                logger.info(
                    f"Polling: New Tab Detected: {new_tab.title} ({new_tab.url}). Processing..."
                )
                tasks.append(self._process_new_tab(new_tab))
            else:
                logger.warning(f"Polling: New tab {new_tab.id} missing websocket URL.")

        for closed_ref in event.closed_tabs:
            # Logging moved inside cancellation loop
            logger.info(
                f"Polling: Processed Closed Tab ID {closed_ref.id} ({closed_ref.url}) event."
            )

        for navigated_tab, old_ref in event.navigated_tabs:
            logger.info(
                f"Polling: Navigated Tab Detected: ID {navigated_tab.id} from {old_ref.url} TO {navigated_tab.url}. Processing..."
            )
            if navigated_tab.webSocketDebuggerUrl:
                tasks.append(self._process_new_tab(navigated_tab))
            else:
                logger.warning(f"Polling: Navigated tab {navigated_tab.id} missing websocket URL.")

        if tasks:
            logger.debug(f"Polling: Gathering {len(tasks)} tab processing tasks.")
            await asyncio.gather(*tasks)
            logger.debug("Polling: Finished gathering tab processing tasks.")

    async def _handle_interaction_update(self, ref: TabReference):
        """Callback for interaction updates (e.g., scroll)."""
        logger.debug(f"Interaction update detected: {ref}")
        # Logic from cli.py's _handle_interaction_update will go here
        pass  # Placeholder

    async def _handle_content_fetched(
        self,
        ref: TabReference,
        image: Optional[Image.Image],
        scroll_y: Optional[int],
        dom_string: Optional[str],
    ):
        """Callback when content is fetched after interaction."""
        logger.info(f"Interaction: Content Fetched: Tab {ref.id} ({ref.url}) ScrollY: {scroll_y}")
        # Update internal state
        self._active_tab_ref = ref

        # Log if DOM string was fetched and update state via callback
        active_dom_string = None  # Default to None
        if dom_string:
            logger.info(f"    Fetched DOM (Length: {len(dom_string)}) via interaction.")
            active_dom_string = dom_string
        else:
            logger.warning(f"    DOM string not fetched for {ref.url} after interaction.")

        # Update active tab info in the UI layer via callback
        try:
            await self._on_active_tab_updated(self._active_tab_ref, active_dom_string)
        except Exception as cb_err:
            logger.error(
                f"Error invoking `_on_active_tab_updated` callback: {cb_err}", exc_info=True
            )

        # Update page content in the UI layer via callback
        if ref.html:
            try:
                # Pass the raw HTML to the UI layer
                await self._on_page_content_ready(ref.html)
                logger.debug(
                    f"Invoked `_on_page_content_ready` for tab {ref.id} via interaction fetch."
                )
            except Exception as page_cb_err:
                logger.error(
                    f"Error invoking `_on_page_content_ready` callback for tab {ref.id} via interaction fetch: {page_cb_err}"
                )
        else:
            logger.warning(
                f"    HTML not fetched for {ref.url} via interaction, cannot update page content."
            )

    async def _process_new_tab(self, tab: ChromeTab):
        html = ws = dom_string = None
        final_url = final_title = None
        if not tab.webSocketDebuggerUrl:
            logger.warning(f"Tab {tab.id} missing ws url, cannot process.")
            return
        ws_url = tab.webSocketDebuggerUrl

        try:
            logger.debug(f"Connecting ws: {ws_url}")
            # Ensure imports for websockets, wait_for_page_load, get_final_url_and_title, get_html_via_ws,
            # capture_tab_screenshot, DomService, CdpBrowserExecutor, DOM_STRING_INCLUDE_ATTRIBUTES are present.
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
                # Capture screenshot regardless of DOM fetch success if URL/title are okay
                if final_title:  # Check title as proxy for basic page accessibility
                    await capture_tab_screenshot(ws_url=ws_url, ws_connection=ws)
            else:
                logger.warning(f"Could not get final URL for {tab.id}")
        except Exception as e:
            logger.exception(f"Error processing tab {tab.id} ({tab.url}): {e}")
        finally:
            # Ensure websocket closure
            if ws:
                try:
                    await ws.close()
                except websockets.exceptions.ConnectionClosedOK:
                    pass  # Expected closure
                except Exception as close_err:
                    logger.error(f"Error closing websocket for tab {tab.id}: {close_err}")

        # Update internal state and notify UI layer
        new_tab_ref = None
        active_dom_string = dom_string  # Use the fetched dom_string
        if final_url:
            new_tab_ref = TabReference(
                id=tab.id, url=final_url, title=final_title, html=html, ws_url=ws_url
            )
            self._active_tab_ref = new_tab_ref  # Update internal reference
            logger.info(f"Active tab internally UPDATED to: {tab.id} ({final_url})")
        elif self._active_tab_ref and self._active_tab_ref.id == tab.id:
            # If this tab *was* the active one but processing failed, clear it internally
            logger.warning(
                f"Clearing internal active tab reference for {tab.id} due to processing failure."
            )
            self._active_tab_ref = None
            active_dom_string = None  # Ensure DOM string is also cleared

        # Callbacks to update the UI layer
        try:
            await self._on_active_tab_updated(self._active_tab_ref, active_dom_string)
        except Exception as cb_err:
            logger.error(
                f"Error invoking `_on_active_tab_updated` callback after processing tab {tab.id}: {cb_err}",
                exc_info=True,
            )

        if html:
            try:
                await self._on_page_content_ready(html)
                logger.debug(f"Invoked `_on_page_content_ready` for tab {tab.id}")
            except Exception as page_cb_err:
                logger.error(
                    f"Error invoking `_on_page_content_ready` callback for tab {tab.id}: {page_cb_err}"
                )

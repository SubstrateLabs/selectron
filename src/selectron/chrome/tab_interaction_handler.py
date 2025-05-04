import asyncio
from typing import Awaitable, Callable, List, Optional

import websockets
from PIL import Image

from selectron.chrome.cdp_executor import CdpBrowserExecutor
from selectron.chrome.chrome_cdp import (
    ChromeTab,
    capture_tab_screenshot,
    get_tabs,
    monitor_user_interactions,
)
from selectron.chrome.types import TabReference
from selectron.dom.dom_attributes import DOM_STRING_INCLUDE_ATTRIBUTES
from selectron.dom.dom_service import DomService
from selectron.util.logger import get_logger

InteractionTabUpdateCallback = Callable[
    [TabReference], Awaitable[None]
]  # Called immediately on interaction (no fresh HTML)
ContentFetchedCallback = Callable[
    [TabReference, Optional[Image.Image], Optional[int], Optional[str]], Awaitable[None]
]  # Called after interaction + fetch (with fresh HTML, screenshot, scrollY, and DOM string)
RehighlightCallback = Callable[[TabReference], Awaitable[None]]  # Pass tab context for rehighlight

logger = get_logger(__name__)

DEBOUNCE_DELAY_SECONDS = 0.75  # Time to wait after last interaction before fetching


class TabInteractionHandler:
    """Handles interaction monitoring, debouncing, and fetching for a single tab."""

    def __init__(
        self,
        tab: ChromeTab,
        interaction_callback: InteractionTabUpdateCallback,
        content_fetched_callback: ContentFetchedCallback,
        rehighlight_callback: RehighlightCallback,
    ):
        self.tab = tab
        self.tab_id = tab.id
        self.ws_url = tab.webSocketDebuggerUrl
        self.interaction_callback = interaction_callback
        self.content_fetched_callback = content_fetched_callback
        self.rehighlight_callback = rehighlight_callback

        self._monitor_task: Optional[asyncio.Task] = None
        self._debounce_timer: Optional[asyncio.TimerHandle] = None
        self._fetch_task: Optional[asyncio.Task] = None
        self._rehighlight_debounce_timer: Optional[asyncio.TimerHandle] = None
        self._is_running = False
        self._last_interaction_scroll_y: Optional[int] = None  # Store last scrollY here

    async def start(self):
        """Starts the interaction monitoring loop for the tab."""
        if self._is_running or not self.ws_url:
            if not self.ws_url:
                logger.warning(
                    f"Cannot start interaction monitor for tab {self.tab_id}: missing WebSocket URL."
                )
            return  # Already running or cannot run

        self._is_running = True
        self._monitor_task = asyncio.create_task(self._run_interaction_monitor_loop())
        # Add a callback to clean up if the task finishes unexpectedly
        self._monitor_task.add_done_callback(self._handle_monitor_completion)

    async def stop(self):
        """Stops the interaction monitoring loop and cleans up resources."""
        if not self._is_running:
            return  # Already stopped
        self._is_running = False
        # Cancel monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            # Optionally await cancellation with timeout
            try:
                await asyncio.wait_for(self._monitor_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Expected exceptions
            except Exception as e:
                logger.error(f"Error waiting for monitor task cancellation for {self.tab_id}: {e}")
        self._monitor_task = None

        # Cancel debounce timer
        if self._debounce_timer:
            self._debounce_timer.cancel()
            self._debounce_timer = None

        # Cancel rehighlight timer
        if self._rehighlight_debounce_timer:
            self._rehighlight_debounce_timer.cancel()
            self._rehighlight_debounce_timer = None

        # Cancel fetch task
        if self._fetch_task and not self._fetch_task.done():
            self._fetch_task.cancel()
            try:
                await asyncio.wait_for(self._fetch_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Expected exceptions
            except Exception as e:
                logger.error(f"Error waiting for fetch task cancellation for {self.tab_id}: {e}")
        self._fetch_task = None
        # logger.debug(f"Interaction monitor stopped for tab {self.tab_id}") # Reduced noise

    def _handle_monitor_completion(self, task: asyncio.Task):
        """Callback executed when the monitor task finishes (normally or abnormally)."""
        if self._is_running:  # If it stopped unexpectedly while we thought it was running
            try:
                # Log exception if task failed
                exc = task.exception()
                if exc:
                    logger.error(
                        f"Interaction monitor task for {self.tab_id} failed: {exc}", exc_info=exc
                    )
            except asyncio.CancelledError:
                logger.debug(
                    f"Interaction monitor task for {self.tab_id} was cancelled."
                )  # Normal stop
            except asyncio.InvalidStateError:
                logger.debug(
                    f"Interaction monitor task for {self.tab_id} completion state invalid."
                )  # Shouldn't happen

            # Ensure cleanup happens even if task dies
            asyncio.create_task(self.stop())  # Schedule stop if it wasn't initiated

    async def _run_interaction_monitor_loop(self):
        """The actual monitoring loop for the tab's interactions via WebSocket."""
        if not self.ws_url:
            return  # Guard

        try:
            async for event in monitor_user_interactions(self.ws_url):
                if not self._is_running:
                    logger.debug(
                        f"Interaction monitoring stopped for tab {self.tab_id}, exiting loop."
                    )
                    break

                # Store scrollY if it's a scroll event
                if event.get("type") == "scroll":
                    scroll_y = event.get("data", {}).get("scrollY")
                    if isinstance(scroll_y, int):
                        self._last_interaction_scroll_y = scroll_y
                        self._handle_scroll_event()  # Trigger rehighlight debounce
                elif event.get("type") == "click":
                    # Reset scrollY on click? Or keep the last known one?
                    # Let's keep the last known one for now, might be relevant if click triggers fetch
                    # logger.debug(f"Tab {self.tab_id} click detected.")
                    pass

                # Create a TabReference for the callback using the initially known tab info
                interaction_tab_ref = TabReference(
                    id=self.tab.id,
                    url=self.tab.url,
                    title=self.tab.title,
                    html=None,  # HTML is not relevant for the interaction *trigger* callback
                    ws_url=self.ws_url,  # Include the websocket URL
                )
                # Call the originally provided callback (e.g., to immediately indicate activity)
                # Schedule the callback correctly, handling both sync and async cases
                if asyncio.iscoroutinefunction(self.interaction_callback):
                    # Use create_task for the immediate interaction signal
                    asyncio.create_task(self.interaction_callback(interaction_tab_ref))
                else:
                    # Run sync callback directly (might block loop briefly if slow)
                    try:
                        # Although InteractionTabUpdateCallback is async, handle potential sync case defensively
                        result = self.interaction_callback(interaction_tab_ref)
                        if asyncio.iscoroutine(result):
                            asyncio.create_task(result)  # If it returns a coroutine, run it
                    except Exception as sync_cb_exc:
                        logger.error(
                            f"Error executing sync interaction callback for tab {self.tab_id}: {sync_cb_exc}",
                            exc_info=True,
                        )

                # Also, handle debouncing to fetch content later
                self._handle_interaction_event()  # Triggers debounce timer reset

        except websockets.exceptions.ConnectionClosedOK:
            logger.debug(f"ws connection closed normally for tab {self.tab_id}.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"ws connection closed with error for tab {self.tab_id}: {e}")
        except websockets.exceptions.InvalidStatus as e:
            # Handle cases where the tab vanished
            # Use getattr for safety, although status_code should exist
            status_code = getattr(e, "status_code", None)
            if status_code == 500 and "No such target id" in str(e):
                logger.warning(
                    f"Interaction monitor for {self.tab_id} ({self.ws_url}) failed: Target ID disappeared (HTTP 500). Stopping handler."
                )
                # Task completion callback will handle stop()
            else:
                logger.error(
                    f"Unexpected InvalidStatus in interaction monitor for tab {self.tab_id}: {e}",
                    exc_info=True,
                )
        except Exception as e:
            logger.error(
                f"Error in interaction monitor loop for tab {self.tab_id}: {e}", exc_info=True
            )
        finally:
            pass
            # logger.debug(f"Interaction monitor loop finished for tab {self.tab_id}")
            # Let the done callback handle cleanup

    def _handle_interaction_event(self):
        """Handles a detected interaction event by resetting the debounce timer."""
        # Cancel existing timer, if any
        if self._debounce_timer:
            self._debounce_timer.cancel()

        # Schedule the debounced fetch function
        loop = asyncio.get_running_loop()
        self._debounce_timer = loop.call_later(
            DEBOUNCE_DELAY_SECONDS,
            lambda: asyncio.create_task(self._trigger_debounced_fetch()),
        )

    def _handle_scroll_event(self):  # New method
        """Handles scroll event by resetting the rehighlight debounce timer."""
        # Cancel existing rehighlight timer, if any
        if self._rehighlight_debounce_timer:
            self._rehighlight_debounce_timer.cancel()

        # Schedule the rehighlight callback
        loop = asyncio.get_running_loop()
        debounce_delay = 0.25  # Shorter debounce for rehighlighting
        self._rehighlight_debounce_timer = loop.call_later(
            debounce_delay,
            lambda: asyncio.create_task(self._invoke_rehighlight_callback()),
        )

    async def _invoke_rehighlight_callback(self):
        """Wrapper to log and invoke the rehighlight callback."""
        try:
            # Create a TabReference from self.tab for the callback
            tab_ref = TabReference(
                id=self.tab.id,
                url=self.tab.url,
                title=self.tab.title,
                ws_url=self.ws_url,
                html=None,  # HTML not needed for rehighlight trigger
            )
            await self.rehighlight_callback(tab_ref)
        except Exception as e:
            logger.error(
                f"Tab {self.tab_id}: Error invoking rehighlight callback: {e}", exc_info=True
            )

    async def _trigger_debounced_fetch(self):
        """Callback executed after the debounce delay. Starts the HTML fetch task."""
        self._debounce_timer = None  # Timer has fired
        # Prevent concurrent fetches for the same tab initiated by interactions
        if self._fetch_task and not self._fetch_task.done():
            logger.debug(
                f"Fetch already in progress for tab {self.tab_id}, skipping debounced trigger."
            )
            return

        # Run the fetch in the background
        self._fetch_task = asyncio.create_task(self._fetch_and_process_tab_content())

        # Ensure the task reference is cleared once it completes
        self._fetch_task.add_done_callback(lambda _task: setattr(self, "_fetch_task", None))

    async def trigger_immediate_fetch(self):
        """Triggers an immediate fetch of tab content, bypassing the debounce timer.
        Skips if a fetch is already in progress.
        """
        # Prevent concurrent fetches for the same tab
        if self._fetch_task and not self._fetch_task.done():
            logger.debug(
                f"Fetch already in progress for tab {self.tab_id}, skipping immediate trigger."
            )
            return

        logger.debug(f"Triggering immediate fetch for tab {self.tab_id}")
        # Run the fetch in the background
        self._fetch_task = asyncio.create_task(self._fetch_and_process_tab_content())
        # Ensure the task reference is cleared once it completes
        self._fetch_task.add_done_callback(lambda _task: setattr(self, "_fetch_task", None))

    async def _fetch_and_process_tab_content(self):
        """Fetches HTML, screenshot, and DOM for the tab and calls the content_fetched_callback."""
        html_content: Optional[str] = None
        screenshot_pil_image: Optional[Image.Image] = None
        dom_string: Optional[str] = None  # Add variable for DOM string
        fetched_tab_ref: Optional[TabReference] = None
        scroll_y_at_capture: Optional[int] = None
        ws = None

        try:
            # Get the current tab info first
            current_tabs: List[ChromeTab] = await get_tabs()
            target_tab_obj = next((t for t in current_tabs if t.id == self.tab_id), None)

            if not target_tab_obj:
                logger.warning(
                    f"Could not find tab {self.tab_id} info for fetching. Aborting update."
                )
                return
            if not target_tab_obj.webSocketDebuggerUrl:
                logger.warning(
                    f"Tab {self.tab_id} has no websocket URL. Cannot fetch/screenshot/dom. Aborting update."
                )
                return

            ws_url = target_tab_obj.webSocketDebuggerUrl
            current_url = target_tab_obj.url  # Get latest URL
            latest_title = target_tab_obj.title  # Get latest title

            # --- Connect to WebSocket --- Need connection for multiple commands
            ws = await websockets.connect(
                ws_url, max_size=30 * 1024 * 1024, open_timeout=10, close_timeout=10
            )

            # --- Instantiate CDP Executor with existing connection --- #
            # Use the executor to ensure Runtime.enable is called if needed
            # No need for async with as we manage ws manually here
            browser_executor = CdpBrowserExecutor(ws_url, current_url, ws_connection=ws)

            # --- Fetch HTML --- #
            try:
                html_script = "document.documentElement.outerHTML"
                html_content = await browser_executor.evaluate(html_script)
                if not html_content:
                    logger.warning(f"Failed to fetch HTML via executor for {self.tab_id}")
            except Exception as html_e:
                logger.error(
                    f"Error fetching HTML via executor for {self.tab_id}: {html_e}", exc_info=True
                )

            # --- Fetch DOM State --- #
            if html_content:  # Only try getting DOM if we have HTML
                try:
                    # Create DomService instance
                    dom_service = DomService(browser_executor)
                    # Get DOM state
                    dom_state = await dom_service.get_elements()
                    # Serialize the DOM tree
                    if dom_state and dom_state.element_tree:
                        # Generate DOM string WITH attributes
                        dom_string = dom_state.element_tree.elements_to_string(
                            include_attributes=DOM_STRING_INCLUDE_ATTRIBUTES
                        )
                        if not dom_string or len(dom_string) < 100:
                            logger.warning(
                                f"DOM string is missing or too short for {self.tab_id}: {dom_string[:100] if dom_string else 'None'}"
                            )
                    else:
                        logger.warning(f"get_elements returned empty state for {self.tab_id}")
                except Exception as dom_e:
                    logger.error(
                        f"Error fetching or serializing DOM for {self.tab_id}: {dom_e}",
                        exc_info=True,
                    )
            else:
                logger.warning(f"Skipping DOM fetch for {self.tab_id} because HTML fetch failed.")

            # --- Capture Screenshot (and get scrollY just before) --- #
            try:
                # Get scrollY immediately before screenshot using the executor
                scroll_script = "window.scrollY"
                scroll_eval = await browser_executor.evaluate(scroll_script)
                if isinstance(scroll_eval, (int, float)):
                    scroll_y_at_capture = int(scroll_eval)
                else:
                    logger.warning(
                        f"Could not get scrollY before screenshot for {self.tab_id}. Fallback: {self._last_interaction_scroll_y}"
                    )
                    scroll_y_at_capture = self._last_interaction_scroll_y  # Fallback

                # Capture screenshot using the *same ws connection* passed to cdp func
                screenshot_pil_image = await capture_tab_screenshot(ws_url=ws_url, ws_connection=ws)
                if not screenshot_pil_image:
                    logger.warning(f"Could not capture screenshot for {self.tab_id}.")
            except Exception as ss_e:
                logger.error(f"Error capturing screenshot for {self.tab_id}: {ss_e}", exc_info=True)

            # --- Create TabReference --- #
            if current_url and self.tab_id:
                fetched_tab_ref = TabReference(
                    id=self.tab_id,
                    url=current_url,
                    html=html_content,
                    title=latest_title,
                    ws_url=ws_url,
                )
            else:
                logger.error(
                    f"Cannot create TabReference for {self.tab_id} due to missing ID or URL."
                )
                return

            # --- Call the callback with reference, image, scrollY, and DOM string --- #
            if asyncio.iscoroutinefunction(self.content_fetched_callback):
                asyncio.create_task(
                    self.content_fetched_callback(
                        fetched_tab_ref, screenshot_pil_image, scroll_y_at_capture, dom_string
                    )
                )
            else:
                try:
                    result = self.content_fetched_callback(
                        fetched_tab_ref, screenshot_pil_image, scroll_y_at_capture, dom_string
                    )
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as sync_cb_exc:
                    logger.error(
                        f"Error executing sync content_fetched_callback for tab {self.tab_id}: {sync_cb_exc}",
                        exc_info=True,
                    )

        except Exception as e:
            logger.error(
                f"Error fetching/processing tab {self.tab_id} after interaction: {e}", exc_info=True
            )
        finally:
            if ws and ws.state != websockets.protocol.State.CLOSED:
                await ws.close()

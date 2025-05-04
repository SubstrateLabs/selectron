import asyncio
import time
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from selectron.chrome.chrome_cdp import ChromeTab, get_tabs
from selectron.chrome.diff_tabs import diff_tabs
from selectron.chrome.tab_interaction_handler import (
    DEBOUNCE_DELAY_SECONDS,
    ContentFetchedCallback,
    InteractionTabUpdateCallback,
    RehighlightCallback,
    TabInteractionHandler,
)
from selectron.chrome.types import TabReference
from selectron.util.logger import get_logger

logger = get_logger(__name__)


class TabChangeEvent(NamedTuple):
    new_tabs: List[ChromeTab]
    closed_tabs: List[TabReference]
    navigated_tabs: List[Tuple[ChromeTab, TabReference]]
    current_tabs: List[ChromeTab]


# Type aliases
PollingTabChangeCallback = Callable[[TabChangeEvent], Awaitable[None]]


class ChromeMonitor:
    """watches Chrome for tab changes (new, closed, navigated) via polling and interactions."""

    def __init__(
        self,
        rehighlight_callback: RehighlightCallback,
        check_interval: float = 2.0,
        interaction_debounce: float = DEBOUNCE_DELAY_SECONDS,
    ):
        """
        Args:
            rehighlight_callback: Callback for rehighlighting
            check_interval: How often to check for tab changes via polling, in seconds
            interaction_debounce: The debounce delay for interaction signals, in seconds
        """
        self.rehighlight_callback = rehighlight_callback
        self.check_interval = check_interval
        self.interaction_debounce = interaction_debounce
        self.previous_tab_refs: Set[TabReference] = set()
        self.last_tabs_check = 0
        self._monitoring = False
        self._on_polling_change_callback: Optional[PollingTabChangeCallback] = None
        self._on_interaction_update_callback: Optional[InteractionTabUpdateCallback] = None
        self._on_content_fetched_callback: Optional[ContentFetchedCallback] = None
        self._monitor_task: Optional[asyncio.Task] = None

        self._interaction_handlers: Dict[str, TabInteractionHandler] = {}

    async def start_monitoring(
        self,
        on_polling_change_callback: PollingTabChangeCallback,
        on_interaction_update_callback: InteractionTabUpdateCallback,
        on_content_fetched_callback: ContentFetchedCallback,
    ) -> bool:
        """
        Start monitoring tabs for changes asynchronously via polling and interactions.
        This method initializes the monitoring process and should only be called once.

        Args:
            on_polling_change_callback: Callback for new/closed/navigated tabs detected by polling.
            on_interaction_update_callback: Callback for immediate interaction signal.
            on_content_fetched_callback: Callback after interaction + debounced content fetch.

        Returns:
            bool: True if monitoring started successfully
        """
        if self._monitoring:
            logger.warning("Tab monitoring start requested, but already running.")
            return False

        self._on_polling_change_callback = on_polling_change_callback
        self._on_interaction_update_callback = on_interaction_update_callback
        self._on_content_fetched_callback = on_content_fetched_callback

        await self._initialize_tabs_and_monitors()

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        return True

    async def _initialize_tabs_and_monitors(self):
        """Gets current tabs, updates state, and starts interaction monitors."""
        try:
            await self._stop_all_interaction_monitors()

            initial_cdp_tabs: List[ChromeTab] = await get_tabs()
            filtered_tabs = [
                tab
                for tab in initial_cdp_tabs
                if tab.webSocketDebuggerUrl
                and tab.id
                and tab.url
                and (tab.url.startswith("http://") or tab.url.startswith("https://"))
            ]

            new_tab_refs = set()
            for tab in filtered_tabs:
                tab_ref = TabReference(
                    id=tab.id,
                    url=tab.url,
                    title=tab.title,
                    html=None,
                    ws_url=tab.webSocketDebuggerUrl,
                )
                new_tab_refs.add(tab_ref)
                await self._start_interaction_monitor(tab)

            self.previous_tab_refs = new_tab_refs

        except Exception as e:
            logger.error(f"Error during initial tab/monitor setup: {e}", exc_info=True)
            self.previous_tab_refs = set()
            await self._stop_all_interaction_monitors()

    async def stop_monitoring(self) -> None:
        """Stop monitoring tabs for changes (async version)."""
        if not self._monitoring:
            logger.debug("Monitoring already stopped.")
            return

        self._monitoring = False
        await self._stop_all_interaction_monitors()
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Main polling task did not stop within timeout.")
            except asyncio.CancelledError:
                pass
                # logger.debug("Main polling task cancelled as expected.")
            except Exception as e:
                logger.error(f"Error stopping main polling task: {e}")

        self.previous_tab_refs = set()
        self._on_polling_change_callback = None
        self._on_interaction_update_callback = None
        self._on_content_fetched_callback = None
        self._monitor_task = None

    # --- Interaction Monitoring Logic --- #
    async def _start_interaction_monitor(self, tab: ChromeTab):
        """Creates and starts a TabInteractionHandler for a single tab."""
        tab_id = tab.id
        ws_url = tab.webSocketDebuggerUrl

        # Extra safety checks
        if not tab_id:
            logger.warning(f"Cannot start interaction monitor: Tab is missing an ID. Data: {tab}")
            return
        if not ws_url:
            logger.warning(
                f"Cannot start interaction monitor for tab {tab_id}: missing WebSocket URL."
            )
            return
        if not self._on_interaction_update_callback:
            logger.error(
                f"Cannot start interaction handler for tab {tab_id}: interaction callback not set."
            )
            return
        if not self._on_content_fetched_callback:
            logger.error(
                f"Cannot start interaction handler for tab {tab_id}: content fetched callback not set."
            )
            return

        # Check if already running
        if tab_id in self._interaction_handlers:
            # logger.debug(f"Interaction handler already running for tab {tab_id}, skipping start.")
            return
        # Create and start handler
        handler = TabInteractionHandler(
            tab=tab,  # Pass the full ChromeTab object
            interaction_callback=self._on_interaction_update_callback,
            content_fetched_callback=self._on_content_fetched_callback,
            rehighlight_callback=self.rehighlight_callback,
        )
        self._interaction_handlers[tab_id] = handler
        await handler.start()

    async def _stop_interaction_monitor(self, tab_id: str):
        """Stops the interaction handler and cleans up resources for a single tab."""
        handler = self._interaction_handlers.pop(tab_id, None)
        if handler:
            await handler.stop()

    async def _stop_all_interaction_monitors(self):
        """Stops all active TabInteractionHandler instances."""
        if not self._interaction_handlers:
            return

        num_handlers = len(self._interaction_handlers)

        # Create stop tasks for all handlers
        # Use list comprehension for clarity
        stop_tasks = [handler.stop() for handler in self._interaction_handlers.values()]

        # Wait for all stop tasks to complete (with a timeout)
        if stop_tasks:
            # Use asyncio.gather for potentially better performance and exception handling
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            # Log any errors that occurred during stopping
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Attempt to get the corresponding handler/tab_id for better logging
                    try:
                        handler_id = list(self._interaction_handlers.keys())[i]
                        logger.warning(
                            f"Error stopping interaction handler for tab {handler_id}: {result}"
                        )
                    except IndexError:
                        logger.warning(
                            f"Error stopping an interaction handler (index {i}): {result}"
                        )
        else:
            logger.debug("No active interaction handlers to stop.")

        # Clear the dictionary regardless of stop success/failure
        self._interaction_handlers.clear()
        logger.debug(f"All ({num_handlers}) interaction handlers stopped and cleared.")

    async def _monitor_loop(self) -> None:
        """Main polling loop."""
        while self._monitoring:
            start_time = time.monotonic()
            try:
                current_cdp_tabs: List[ChromeTab] = await get_tabs()
                if not self._monitoring:
                    break

                changed_tabs_event = await self.process_tab_changes(current_cdp_tabs)

                if self._on_polling_change_callback and changed_tabs_event:
                    if asyncio.iscoroutinefunction(self._on_polling_change_callback):
                        await self._on_polling_change_callback(changed_tabs_event)
                    else:
                        self._on_polling_change_callback(changed_tabs_event)

            except Exception as e:
                logger.error(f"Error during polling check in _monitor_loop: {e}", exc_info=True)

            elapsed_time = time.monotonic() - start_time
            sleep_duration = max(0, self.check_interval - elapsed_time)
            await asyncio.sleep(sleep_duration)

    async def process_tab_changes(
        self, current_cdp_tabs: List[ChromeTab]
    ) -> Optional[TabChangeEvent]:
        """
        Process changes based on polled tabs and manage interaction monitors.
        Compares current tabs with previous state, starts/stops monitors, and returns changes.
        Ignores tabs with non-http(s) schemes or missing WebSocket URLs.
        NOTE: Does NOT fetch HTML here. Fetching is triggered by interaction handlers.

        Args:
            current_cdp_tabs: List of current ChromeTab objects from get_tabs()

        Returns:
            TabChangeEvent if polling detected new/closed/navigated tabs, None otherwise.
        """
        filtered_tabs = [
            tab
            for tab in current_cdp_tabs
            if tab.webSocketDebuggerUrl
            and tab.id
            and tab.url
            and (tab.url.startswith("http://") or tab.url.startswith("https://"))
        ]

        current_tab_refs_map: Dict[str, TabReference] = {
            ref.id: ref for ref in self.previous_tab_refs
        }
        current_polled_tabs_map: Dict[str, ChromeTab] = {tab.id: tab for tab in filtered_tabs}

        added_tabs, removed_refs, navigated_pairs = diff_tabs(
            current_polled_tabs_map, current_tab_refs_map
        )

        # Process Added Tabs
        for tab in added_tabs:
            await self._start_interaction_monitor(tab)
            # NOTE: Trigger immediate fetch for newly added tabs as well.
            handler = self._interaction_handlers.get(tab.id)
            if handler:
                logger.debug(f"Polling: Triggering immediate fetch for new tab {tab.id}")
                asyncio.create_task(handler.trigger_immediate_fetch())

        # Process Removed Tabs
        closed_ids = set()  # Keep track for event reporting if needed
        for ref in removed_refs:
            # logger.debug(f"Polling: Stopping monitor for closed tab {ref.id}") # DEBUG
            await self._stop_interaction_monitor(ref.id)
            closed_ids.add(ref.id)

        # Process Navigated Tabs
        for new_tab, _ in navigated_pairs:
            tab_id = new_tab.id
            # logger.debug(f"Polling: Handling navigation for tab {tab_id} to {new_tab.url}") # DEBUG
            # 1. Stop interaction monitor for old context
            await self._stop_interaction_monitor(tab_id)

            # 2. Start interaction monitor for new context
            await self._start_interaction_monitor(new_tab)

            # 3. Trigger initial fetch for navigated tab's new content
            handler = self._interaction_handlers.get(tab_id)  # Use tab_id which is same for new_tab
            if handler:
                logger.debug(f"Polling: Triggering immediate fetch for navigated tab {tab_id}")
                asyncio.create_task(handler.trigger_immediate_fetch())

        # --- Update State ---
        # Create the new set of references directly from the current filtered tabs
        new_tab_refs = set()
        for tab in filtered_tabs:
            # Ensure essential properties exist before creating ref
            if tab.id and tab.url and tab.webSocketDebuggerUrl:
                new_tab_refs.add(
                    TabReference(
                        id=tab.id,
                        url=tab.url,
                        title=tab.title,
                        ws_url=tab.webSocketDebuggerUrl,
                        html=None,  # HTML is fetched on demand by handler
                    )
                )

        # Atomically update the main state
        self.previous_tab_refs = new_tab_refs

        # --- Construct and return event --- #
        polling_detected_changes = bool(added_tabs or removed_refs or navigated_pairs)

        if polling_detected_changes:
            # logger.debug(f"Polling changes detected: +{len(added_tabs)} / -{len(removed_refs)} / ~{len(navigated_pairs)}") # DEBUG
            # Note: removed_refs contains TabReference objects, consistent with original TabChangeEvent expectation
            return TabChangeEvent(
                new_tabs=added_tabs,
                closed_tabs=list(removed_refs),
                navigated_tabs=navigated_pairs,
                current_tabs=filtered_tabs,
            )
        else:
            return None

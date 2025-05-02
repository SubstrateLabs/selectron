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
    Union,
)

from selectron.chrome.chrome_cdp import get_tabs
from selectron.chrome.diff_tabs import diff_tabs
from selectron.chrome.types import ChromeTab, TabReference
from selectron.util.logger import get_logger

logger = get_logger(__name__)


class TabChangeEvent(NamedTuple):
    new_tabs: List[ChromeTab]
    closed_tabs: List[TabReference]
    navigated_tabs: List[Tuple[ChromeTab, TabReference]]
    current_tabs: List[ChromeTab]


PollingTabChangeCallback = Union[
    Callable[[TabChangeEvent], None],
    Callable[[TabChangeEvent], Awaitable[None]],
]


class ChromeMonitor:
    """watches Chrome for tab changes (new, closed, navigated) via polling."""

    def __init__(self, check_interval: float = 2.0):
        """
        Args:
            check_interval: How often to check for tab changes, in seconds
        """
        self.check_interval = check_interval
        self.previous_tab_refs: Set[TabReference] = set()
        self.last_tabs_check = 0
        self._monitoring = False
        self._on_polling_change_callback: Optional[PollingTabChangeCallback] = None
        self._monitor_task = None

    async def start_monitoring(
        self,
        on_polling_change_callback: PollingTabChangeCallback,
    ) -> bool:
        """
        Start monitoring tabs for changes asynchronously via polling.
        This method initializes the monitoring process and should only be called once.

        Args:
            on_polling_change_callback: Callback for new/closed/navigated tabs detected by polling.

        Returns:
            bool: True if monitoring started successfully
        """
        if self._monitoring:
            logger.warning("Tab monitoring start requested, but already running.")
            return False

        self._on_polling_change_callback = on_polling_change_callback

        logger.info("Performing initial tab check...")
        try:
            initial_cdp_tabs: List[ChromeTab] = await get_tabs()
            filtered_initial_tabs = [
                tab
                for tab in initial_cdp_tabs
                if tab.webSocketDebuggerUrl
                and (tab.url.startswith("http://") or tab.url.startswith("https://"))
            ]
            self.previous_tab_refs = {
                TabReference(
                    id=tab.id, url=tab.url, title=tab.title, ws_url=tab.webSocketDebuggerUrl
                )
                for tab in filtered_initial_tabs
                if tab.id and tab.url
            }
            logger.info(
                f"Initial tab check complete. Found {len(self.previous_tab_refs)} relevant tabs."
            )
        except Exception as e:
            logger.error(f"Error during initial tab check: {e}", exc_info=True)
            self.previous_tab_refs = set()

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Chrome tab monitoring started.")
        return True

    async def stop_monitoring(self) -> None:
        """Stop monitoring tabs for changes (async version)."""
        if not self._monitoring:
            logger.debug("Monitoring already stopped.")
            return

        self._monitoring = False

        if self._monitor_task and not self._monitor_task.done():
            logger.debug("Stopping main polling task...")
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=2.0)
                logger.debug("Main polling task cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning("Main polling task did not stop within timeout.")
            except asyncio.CancelledError:
                logger.debug("Main polling task cancelled as expected.")
            except Exception as e:
                logger.error(f"Error stopping main polling task: {e}")

        self.previous_tab_refs = set()
        self._on_polling_change_callback = None
        self._monitor_task = None

        logger.info("Chrome tab monitoring stopped.")

    async def _monitor_loop(self) -> None:
        """Main polling loop."""
        while self._monitoring:
            start_time = time.monotonic()
            try:
                current_cdp_tabs: List[ChromeTab] = await get_tabs()
                if not self._monitoring:
                    break

                changed_tabs_event = self.process_tab_changes(current_cdp_tabs)

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

        logger.info("Exiting ChromeTabs _monitor_loop.")

    def process_tab_changes(self, current_cdp_tabs: List[ChromeTab]) -> Optional[TabChangeEvent]:
        """
        Process changes based on polled tabs.
        Compares current tabs with previous state and returns changes.
        Ignores tabs with non-http(s) schemes or missing WebSocket URLs.

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

        updated_refs = set(self.previous_tab_refs)

        closed_ids = {ref.id for ref in removed_refs}
        updated_refs = {ref for ref in updated_refs if ref.id not in closed_ids}

        for tab in added_tabs:
            updated_refs.add(
                TabReference(
                    id=tab.id, url=tab.url, title=tab.title, ws_url=tab.webSocketDebuggerUrl
                )
            )

        navigated_ids_to_update = {pair[0].id for pair in navigated_pairs}
        updated_refs = {ref for ref in updated_refs if ref.id not in navigated_ids_to_update}
        for new_tab, _old_ref in navigated_pairs:
            updated_refs.add(
                TabReference(
                    id=new_tab.id,
                    url=new_tab.url,
                    title=new_tab.title,
                    ws_url=new_tab.webSocketDebuggerUrl,
                )
            )

        self.previous_tab_refs = updated_refs

        polling_detected_changes = bool(added_tabs or removed_refs or navigated_pairs)

        if polling_detected_changes:
            return TabChangeEvent(
                new_tabs=added_tabs,
                closed_tabs=list(removed_refs),
                navigated_tabs=navigated_pairs,
                current_tabs=filtered_tabs,
            )
        else:
            return None

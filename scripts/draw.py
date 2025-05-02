import asyncio
import logging
import signal
import sys
from typing import Optional

import websockets
from PIL import Image
from rich.console import Console

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
from selectron.util.extract_metadata import HtmlMetadata, extract_metadata
from selectron.util.logger import get_logger
from selectron.util.sample_save import save_sample_data

logger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.INFO)
console = Console()


# --- Helper Functions (moved most to sampler_utils) ---

# --- Event Handlers ---


async def handle_polling_change(event: TabChangeEvent):
    """Callback function for tab changes detected ONLY by polling."""
    tasks = []
    for new_tab in event.new_tabs:
        if new_tab.webSocketDebuggerUrl:
            console.print(
                f"  [cyan]Polling: New Tab Detected:[/cyan] {new_tab.title} ({new_tab.url})"
            )
            tasks.append(process_new_tab(new_tab))
        else:
            logger.warning(
                f"Polling: New tab {new_tab.id} missing websocket URL, skipping initial process."
            )

    for closed_ref in event.closed_tabs:
        console.print(f"  [red]Polling: Closed Tab:[/red] ID {closed_ref.id} ({closed_ref.url})")

    for navigated_tab, old_ref in event.navigated_tabs:
        console.print(
            f"  [yellow]Polling: Navigated Tab:[/yellow] ID {navigated_tab.id} from {old_ref.url} TO {navigated_tab.url} Title: {navigated_tab.title}"
        )
        if navigated_tab.webSocketDebuggerUrl:
            tasks.append(process_new_tab(navigated_tab))
        else:
            logger.warning(
                f"Polling: Navigated tab {navigated_tab.id} missing websocket URL, skipping initial process."
            )

    if tasks:
        await asyncio.gather(*tasks)


async def handle_interaction_update(ref: TabReference):
    pass


async def handle_content_fetched(
    ref: TabReference,
    image: Optional[Image.Image],
    scroll_y: Optional[int],
    dom_string: Optional[str],
):
    """Callback triggered after interaction + debounce + content fetch."""
    console.print(
        f"  [green]Interaction: Content Fetched:[/green] Tab {ref.id} ({ref.title} - {ref.url}) ScrollY: {scroll_y if scroll_y is not None else 'N/A'}"
    )

    # Log if DOM string was fetched
    if dom_string:
        console.print(f"    [blue]Info:[/blue] Fetched DOM (Length: {len(dom_string)})")
    else:
        console.print(
            f"    [yellow]Warning:[/yellow] DOM string not fetched for {ref.url} after interaction."
        )

    if not ref.html:
        console.print(
            f"    [yellow]Warning:[/yellow] HTML not fetched for {ref.url}, skipping metadata/save."
        )
        return

    metadata: Optional[HtmlMetadata] = None
    try:
        metadata = extract_metadata(ref.html, url=ref.url)
    except Exception as e:
        logger.error(
            f"Error extracting metadata for {ref.url} after interaction: {e}", exc_info=True
        )
        console.print(f"    [red]Error:[/red] Failed extracting metadata for {ref.url}: {e}")

    if not image:
        console.print(f"    [yellow]Warning:[/yellow] No screenshot captured for {ref.url}.")

    if ref.html and metadata:
        await save_sample_data(
            url=ref.url,
            html=ref.html,
            metadata=metadata,
            image=image,
            dom_string=dom_string,
        )
    else:
        missing = []
        if not ref.html:
            missing.append("HTML")
        if not metadata:
            missing.append("Metadata")
        if not image:
            missing.append("Image")
        logger.warning(f"Skipping save for {ref.url} due to missing data: {', '.join(missing)}")
        console.print(
            f"    [yellow]Warning:[/yellow] Skipping save for {ref.url} due to missing: {', '.join(missing)}."
        )


async def process_new_tab(tab: ChromeTab):
    """Fetches HTML, metadata, screenshot for a NEW or NAVIGATED tab and saves samples.
    Connects via WebSocket, waits for load, gets final URL, then fetches content.
    """
    html: Optional[str] = None
    metadata: Optional[HtmlMetadata] = None
    screenshot_pil_image: Optional[Image.Image] = None
    ws = None
    dom_string: Optional[str] = None

    if not tab.webSocketDebuggerUrl:
        logger.warning(f"Tab {tab.id} missing websocket URL, cannot process.")
        return

    ws_url = tab.webSocketDebuggerUrl
    final_url: Optional[str] = None
    final_title: Optional[str] = None

    try:
        logger.debug(f"Connecting to WebSocket: {ws_url}")
        ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024)
        logger.debug(f"Connected to WebSocket for tab {tab.id}.")

        loaded = await wait_for_page_load(ws)
        if not loaded:
            logger.warning(f"Page load timeout/failure for {tab.id}, proceeding anyway.")
        else:
            logger.debug(f"Page load confirmed for tab {tab.id}.")

        settle_delay = 1.0
        logger.debug(f"Waiting {settle_delay}s for page {tab.id} to settle...")
        await asyncio.sleep(settle_delay)

        final_url, final_title = await get_final_url_and_title(
            ws, tab.url, tab.title or "Unknown", tab_id_for_logging=tab.id
        )
        if final_url:
            html = await get_html_via_ws(ws, final_url)
        else:
            logger.warning(f"Skipping HTML fetch for {tab.id} because final URL is missing.")
            html = None

        if html and final_url and final_title:
            try:
                metadata = extract_metadata(html, url=final_url)
            except Exception as meta_e:
                logger.error(f"Error extracting metadata for {final_url}: {meta_e}", exc_info=True)
                console.print(
                    f"    [red]Error:[/red] Failed extracting metadata for {final_url}: {meta_e}"
                )
                metadata = None
        else:
            missing_for_meta = []
            if not html:
                missing_for_meta.append("HTML")
            if not final_url:
                missing_for_meta.append("Final URL")
            if not final_title:
                missing_for_meta.append("Final Title")
            logger.warning(
                f"Skipping metadata extraction for {tab.id} due to missing data: {', '.join(missing_for_meta)}"
            )
            metadata = None

        if html and final_url:
            try:
                logger.debug(f"Fetching DOM state via executor for {tab.id}")
                browser_executor = CdpBrowserExecutor(ws_url, final_url, ws_connection=ws)
                dom_service = DomService(browser_executor)
                dom_state = await dom_service.get_elements()
                if dom_state and dom_state.element_tree:
                    dom_string = dom_state.element_tree.elements_to_string(
                        include_attributes=DOM_STRING_INCLUDE_ATTRIBUTES
                    )
                else:
                    console.print(
                        f"    [yellow]Warning:[/yellow] get_elements returned empty state for {final_url}"
                    )
                    dom_string = None
            except Exception as dom_e:
                logger.error(
                    f"Error fetching/serializing DOM for {final_url}: {dom_e}", exc_info=True
                )
                console.print(
                    f"    [red]Error:[/red] Failed fetching/serializing DOM for {final_url}: {dom_e}"
                )
                dom_string = None
        else:
            logger.warning(f"Skipping DOM fetch for {tab.id} because HTML or final URL is missing.")
            dom_string = None

        if final_url and final_title:
            try:
                screenshot_pil_image = await capture_tab_screenshot(ws_url=ws_url, ws_connection=ws)
            except Exception as ss_e:
                logger.error(f"Error capturing screenshot for {final_url}: {ss_e}", exc_info=True)
                console.print(
                    f"    [red]Error:[/red] Failed capturing screenshot for {final_url}: {ss_e}"
                )
                screenshot_pil_image = None
        else:
            logger.warning(
                f"Skipping screenshot capture for {tab.id} because final URL or Title is missing."
            )
            screenshot_pil_image = None

    except websockets.exceptions.WebSocketException as ws_e:
        logger.error(f"WebSocket error during processing tab {tab.id}: {ws_e}", exc_info=True)
        console.print(f"    [red]Error:[/red] WebSocket failed for {tab.url}: {ws_e}")
    except Exception as e:
        logger.error(f"Unexpected error processing tab {tab.id} ({tab.url}): {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Unexpected error for {tab.url}: {e}")
    finally:
        if ws and ws.state != websockets.protocol.State.CLOSED:
            try:
                await ws.close()
                logger.debug(f"WebSocket closed for tab {tab.id}.")
            except Exception as close_e:
                logger.warning(f"Error closing WebSocket for {tab.id}: {close_e}")

    # --- Logging Summary ---
    fetched_summary = []
    if html:
        fetched_summary.append("HTML")
    if dom_string:
        fetched_summary.append("DOM")
    if screenshot_pil_image:
        fetched_summary.append("Screenshot")

    log_url_for_summary = final_url if final_url else tab.url
    log_title_for_summary = final_title if final_title else tab.title

    if final_url:
        status_color = "green" if html and metadata else "yellow"
        console.print(
            f"  [{status_color}]Navigation: Content Processed:[/{status_color}] Tab {tab.id} ({log_title_for_summary} - {log_url_for_summary}) Fetched: {', '.join(fetched_summary) if fetched_summary else 'None'}"
        )
    else:
        console.print(
            f"  [red]Navigation: Processing Failed:[/red] Tab {tab.id} ({tab.url}) - Could not determine final URL/Title."
        )

    # --- Save Data ---
    if html and metadata and final_url:
        await save_sample_data(
            url=final_url,
            html=html,
            metadata=metadata,
            image=screenshot_pil_image,
            dom_string=dom_string,
        )
    else:
        missing = []
        if not html:
            missing.append("HTML")
        if not metadata:
            missing.append("Metadata")
        if not final_url:
            missing.append("Final URL")

        log_url_for_warning = final_url if final_url else tab.url
        logger.warning(
            f"Skipping save for {log_url_for_warning} due to missing data: {', '.join(missing)}"
        )


async def main():
    if not await ensure_chrome_connection():
        console.print("[bold red]Failed to establish Chrome connection. Exiting.[/bold red]")
        sys.exit(1)

    monitor = ChromeMonitor(check_interval=1.5)
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        print()  # Add a newline after ^C
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    wait_task = None
    monitor = None  # Initialize monitor to None

    try:
        console.print("[cyan]Starting Chrome monitor...[/cyan]")
        # Initialize monitor here
        monitor = ChromeMonitor(check_interval=1.5)
        # Pass callbacks to start_monitoring
        monitor_started = await monitor.start_monitoring(
            on_polling_change_callback=handle_polling_change,
            on_interaction_update_callback=handle_interaction_update,
            on_content_fetched_callback=handle_content_fetched,
        )

        if not monitor_started:
            console.print("[bold red]Failed to start monitor. Exiting.[/bold red]")
            return  # No need for sys.exit if returning from main

        console.print("[green]Monitor started successfully.[/green]")

        # Create only the shutdown wait task
        wait_task = asyncio.create_task(shutdown_event.wait(), name="ShutdownWaiter")

        # Wait for the shutdown signal
        await wait_task

        # Check results of completed tasks for errors (only wait_task in this case)
        try:
            wait_task.result()  # Raise exception if task failed
        except asyncio.CancelledError:
            logger.info(f"Task {wait_task.get_name()} was cancelled gracefully.")
        except Exception as e:
            logger.error(f"Task {wait_task.get_name()} failed unexpectedly: {e}", exc_info=True)
            console.print(f"[red]Error in {wait_task.get_name()}: {e}[/red]")
            # Ensure shutdown if a task fails unexpectedly
            shutdown_event.set()

    except Exception as e:
        logger.error(f"An unexpected error occurred in main setup/wait: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        shutdown_event.set()  # Ensure shutdown on main error
    finally:
        console.print("[cyan]Shutting down...[/cyan]")
        # Ensure event is set if not already (e.g., loop exited normally but signal wasn't caught)
        if not shutdown_event.is_set():
            logger.info("Shutdown event not set, setting it now in finally block.")
            shutdown_event.set()

        # Cancel any remaining pending tasks (should only be wait_task if it was cancelled)
        tasks_to_cancel = []
        if wait_task and not wait_task.done():
            tasks_to_cancel.append(wait_task)

        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} pending tasks...")
            for task in tasks_to_cancel:
                task.cancel()

            # Gather cancelled tasks to allow cleanup
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info("Gathered cancelled pending tasks.")
        else:
            logger.info("No pending tasks needed cancellation.")

        # Stop the monitor
        if monitor:  # Ensure monitor object was successfully created
            console.print("[cyan]Stopping Chrome monitor (if running)...[/cyan]")
            await monitor.stop_monitoring()  # stop_monitoring handles internal checks
            console.print("[green]Monitor stop process completed.[/green]")
        else:
            logger.info("Monitor object was not initialized, skipping stop.")

        console.print("[bold green]Shutdown complete.[/bold green]")


if __name__ == "__main__":
    # Consider adding basic logging setup here if needed earlier
    # logging.basicConfig(level=logging.DEBUG) # Example
    asyncio.run(main())

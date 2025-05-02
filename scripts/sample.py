import asyncio
import logging
import signal
import sys
from typing import Optional

import websockets
from PIL import Image
from rich.console import Console
from rich.pretty import pretty_repr
from term_image.exceptions import InvalidSizeError
from term_image.image import AutoImage

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
from selectron.sampler.sampler_utils import save_sample_data
from selectron.util.extract_metadata import HtmlMetadata, extract_metadata
from selectron.util.logger import get_logger

logger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.INFO)
console = Console()


# --- Helper Functions (moved most to sampler_utils) ---

# --- Event Handlers ---


async def handle_polling_change(event: TabChangeEvent):
    """Callback function for tab changes detected ONLY by polling."""
    console.print(
        f"\n[bold blue]Polling Event:[/bold blue] New: {len(event.new_tabs)}, Closed: {len(event.closed_tabs)}, Navigated: {len(event.navigated_tabs)}"
    )

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
    """Callback triggered immediately on user interaction (click/scroll)."""
    console.print(
        f"  [magenta]Interaction:[/magenta] User action detected in tab {ref.id} ({ref.title} - {ref.url})"
    )


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
        if image:
            _display_screenshot(image, ref.title or "Unknown Title")
        return

    metadata: Optional[HtmlMetadata] = None
    try:
        metadata = extract_metadata(ref.html, url=ref.url)
        console.print(f"    [green]Success:[/green] Extracted Metadata for {ref.title}")
        logger.info(
            f"Extracted metadata for {ref.url}: {pretty_repr(metadata, indent_size=2, max_length=20)}"
        )
    except Exception as e:
        logger.error(
            f"Error extracting metadata for {ref.url} after interaction: {e}", exc_info=True
        )
        console.print(f"    [red]Error:[/red] Failed extracting metadata for {ref.url}: {e}")

    if image:
        _display_screenshot(image, ref.title or "Unknown Title")
    else:
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


def _display_screenshot(image: Image.Image, title: str):
    console.print(
        f"    [green]Success:[/green] Captured screenshot for {title}. Displaying cropped preview:"
    )
    try:
        # Crop image for terminal display (max height 600px)
        width, height = image.size
        crop_height = min(height, 600)
        cropped_image = image.crop((0, 0, width, crop_height))

        # Use the cropped image for terminal display
        term_img = AutoImage(cropped_image, width=80)
        term_img.draw()
    except InvalidSizeError:
        console.print(
            "    [yellow]Warning:[/yellow] Cropped screenshot is still too large to display in the terminal."
        )
    except Exception as draw_e:
        logger.error(f"Error drawing screenshot: {draw_e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Failed to display screenshot: {draw_e}")


async def process_new_tab(tab: ChromeTab):
    """Fetches HTML, metadata, screenshot for a NEW or NAVIGATED tab and saves samples.
    Connects via WebSocket, waits for load, gets final URL, then fetches content.
    """
    logger.info(f"Processing tab {tab.id}: {tab.title} ({tab.url})")
    console.print(f"    [cyan]Processing Tab:[/cyan] {tab.title} ({tab.url})")
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
        logger.info(f"Tab {tab.id} final URL: {final_url}, Title: {final_title}")

        if final_url:
            html = await get_html_via_ws(ws, final_url)
        else:
            logger.warning(f"Skipping HTML fetch for {tab.id} because final URL is missing.")
            html = None

        if html and final_url and final_title:
            try:
                metadata = extract_metadata(html, url=final_url)
                console.print(f"    [green]Success:[/green] Extracted Metadata for {final_title}")
                logger.info(
                    f"Extracted metadata for {final_url}: {pretty_repr(metadata, indent_size=2, max_length=20)}"
                )
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
                dom_state = await dom_service.get_clickable_elements(highlight_elements=False)
                if dom_state and dom_state.element_tree:
                    dom_string = dom_state.element_tree.clickable_elements_to_string(
                        include_attributes=DOM_STRING_INCLUDE_ATTRIBUTES
                    )
                    console.print(
                        f"    [green]Success:[/green] Fetched and serialized DOM for {final_url} (Length: {len(dom_string)})"
                    )
                else:
                    console.print(
                        f"    [yellow]Warning:[/yellow] get_clickable_elements returned empty state for {final_url}"
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
                if screenshot_pil_image:
                    _display_screenshot(screenshot_pil_image, final_title)
                else:
                    console.print(
                        f"    [yellow]Warning:[/yellow] Could not capture screenshot for {final_url}."
                    )
                    screenshot_pil_image = None
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

    if not final_url:
        logger.error(
            f"Cannot save data for tab {tab.id} because final URL could not be determined."
        )
        return

    if html and metadata:
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
        if not screenshot_pil_image:
            missing.append("Image")
        logger.warning(f"Skipping save for {final_url} due to missing data: {', '.join(missing)}")
        console.print(
            f"    [yellow]Warning:[/yellow] Skipping save for {final_url} due to missing: {', '.join(missing)}."
        )


async def main():
    if not await ensure_chrome_connection():
        console.print("[bold red]Failed to establish Chrome connection. Exiting.[/bold red]")
        sys.exit(1)

    monitor = ChromeMonitor(check_interval=1.5)  # Create monitor instance
    shutdown_event = asyncio.Event()  # Event to signal shutdown

    def signal_handler(sig, frame):
        console.print(f"\n[bold yellow]Received signal {sig}, initiating shutdown...[/bold yellow]")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console.print("[bold green]Starting Chrome Monitor... Press Ctrl+C to exit.[/bold green]")
    try:
        # Start monitoring in the background, passing all three callbacks
        monitor_started = await monitor.start_monitoring(
            on_polling_change_callback=handle_polling_change,
            on_interaction_update_callback=handle_interaction_update,
            on_content_fetched_callback=handle_content_fetched,
        )

        if not monitor_started:
            console.print("[bold red]Failed to start monitor. Exiting.[/bold red]")
            return

        # Wait indefinitely until shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        console.print("\n[bold yellow]Stopping Chrome Monitor...[/bold yellow]")
        await monitor.stop_monitoring()
        console.print("[bold green]Monitor stopped. Exiting.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())

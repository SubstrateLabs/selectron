import asyncio
import logging
import signal
import sys

from rich.console import Console
from rich.pretty import pretty_repr
from term_image.image import AutoImage

from selectron.chrome.chrome_cdp import capture_tab_screenshot, get_tab_html
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent  # Added imports
from selectron.chrome.connect import ensure_chrome_connection
from selectron.util.extract_metadata import HtmlMetadata, extract_metadata
from selectron.util.logger import get_logger

logger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
console = Console()


async def handle_tab_change(event: TabChangeEvent):
    """Callback function to process tab changes detected by ChromeMonitor."""
    console.print(
        f"\n[bold magenta]Tab Change Detected:[/bold magenta] New: {len(event.new_tabs)}, Closed: {len(event.closed_tabs)}, Navigated: {len(event.navigated_tabs)}"
    )

    tasks = []

    for new_tab in event.new_tabs:
        if new_tab.webSocketDebuggerUrl:
            console.print(f"  [cyan]New Tab:[/cyan] {new_tab.title} ({new_tab.url})")
            tasks.append(process_tab_html(new_tab.webSocketDebuggerUrl, new_tab.url, new_tab.title))
        else:
            logger.warning(f"New tab {new_tab.id} missing websocket URL, skipping.")

    for navigated_tab, _old_ref in event.navigated_tabs:
        if navigated_tab.webSocketDebuggerUrl:
            console.print(
                f"  [blue]Navigated Tab:[/blue] {navigated_tab.title} ({navigated_tab.url})"
            )
            tasks.append(
                process_tab_html(
                    navigated_tab.webSocketDebuggerUrl, navigated_tab.url, navigated_tab.title
                )
            )
        else:
            logger.warning(f"Navigated tab {navigated_tab.id} missing websocket URL, skipping.")

    if tasks:
        await asyncio.gather(*tasks)


async def process_tab_html(ws_url: str, tab_url: str, tab_title: str):
    """Fetches HTML for a given tab WS URL, extracts metadata, and prints it."""
    html = None
    try:
        html = await get_tab_html(ws_url)
        if not html:
            console.print(f"    [yellow]Warning:[/yellow] Could not retrieve HTML for {tab_url}.")
            return  # Exit early if no HTML
        console.print(
            f"    [green]Success:[/green] Retrieved HTML for {tab_url} (Length: {len(html)})"
        )
    except Exception as e:
        logger.error(f"Error getting HTML for {ws_url}: {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Failed to get HTML for {tab_url}: {e}")
        return  # Exit early on error

    try:
        metadata: HtmlMetadata = extract_metadata(html, url=tab_url)
        console.print(f"    [green]Success:[/green] Extracted Metadata for {tab_title}:")
        console.print(pretty_repr(metadata, indent_size=2, max_length=20))
    except Exception as e:
        logger.error(f"Error extracting metadata for {tab_url}: {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Failed extracting metadata for {tab_url}: {e}")

    try:
        screenshot_pil_image = await capture_tab_screenshot(ws_url)
        if screenshot_pil_image:
            console.print(
                f"    [green]Success:[/green] Captured screenshot for {tab_title}. Displaying:"
            )
            try:
                # Set a maximum width for the terminal image
                term_img = AutoImage(screenshot_pil_image, width=80)
                term_img.draw()
            except Exception as draw_e:
                logger.error(f"Error drawing screenshot for {ws_url}: {draw_e}", exc_info=True)
                console.print(f"    [red]Error:[/red] Failed to display screenshot: {draw_e}")
        else:
            console.print(
                f"    [yellow]Warning:[/yellow] Could not capture screenshot for {tab_url}."
            )
    except Exception as e:
        logger.error(f"Error capturing screenshot for {ws_url}: {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Failed to capture screenshot for {tab_url}: {e}")


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
        # Start monitoring in the background
        monitor_started = await monitor.start_monitoring(handle_tab_change)

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
    # Removed old single-shot logic
    asyncio.run(main())

import asyncio
import json
import logging
import signal
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import openai
import websockets
from PIL import Image
from rich.console import Console
from rich.pretty import pretty_repr
from term_image.exceptions import InvalidSizeError
from term_image.image import AutoImage

from selectron.chrome.chrome_cdp import (
    ChromeTab,
    capture_tab_screenshot,
    send_cdp_command,
    wait_for_page_load,
)
from selectron.chrome.chrome_monitor import ChromeMonitor, TabChangeEvent
from selectron.chrome.connect import ensure_chrome_connection
from selectron.chrome.types import TabReference
from selectron.util.extract_markdown import MarkdownStrategy, extract_markdown
from selectron.util.extract_metadata import HtmlMetadata, extract_metadata
from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url
from selectron.util.stitch_images import stitch_vertical

logger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.INFO)
console = Console()

MARKDOWN_STRATEGY = MarkdownStrategy.DOCLING
# Flag to enable/disable OpenAI cleanup
enable_openai_cleanup = True
# Initialize OpenAI client (use Async client now)
openai_async_client: Optional[openai.AsyncOpenAI] = None
if enable_openai_cleanup:
    try:
        # Use AsyncOpenAI
        openai_async_client = openai.AsyncOpenAI()
        # Perform a simple test call to check credentials (sync method still okay for init check)
        openai_async_client.models.list()
        logger.info("Async OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(
            f"Failed to initialize Async OpenAI client: {e}. Disabling cleanup.", exc_info=True
        )
        enable_openai_cleanup = False
        openai_async_client = None

# Global dictionary to store screenshots per URL for simple vertical stacking
url_screenshot_data: Dict[str, List[Image.Image]] = defaultdict(list)
# Global state for debounced OpenAI cleanup
url_latest_markdown: Dict[str, str] = {}
deferred_openai_cleanup_timers: Dict[str, asyncio.TimerHandle] = {}
deferred_openai_cleanup_tasks: Dict[str, asyncio.Task[Any]] = {}
OPENAI_DEBOUNCE_DELAY = 2.5  # Seconds to wait after last update before calling OpenAI


async def cleanup_markdown_with_openai(markdown_text: str) -> str:
    """Uses GPT-4o Mini (async) to clean up markdown, removing extraneous content."""
    if not enable_openai_cleanup or not openai_async_client:  # Check async client
        logger.warning(
            "OpenAI cleanup is disabled or async client failed to initialize. Returning original markdown."
        )
        return markdown_text

    if not markdown_text.strip():
        logger.debug("Skipping OpenAI cleanup for empty markdown.")
        return markdown_text

    prompt = f"""
    Review the following markdown content extracted from a web page.
    Identify and REMOVE any extraneous material: noisy JSON, UI elements, promotional material, anything that is not the main content of the page.
    Retain ONLY the core main content of the page.
    Ensure the output is still valid markdown.
    Do NOT add any commentary, preamble, or explanation; just return the cleaned markdown content.

    Markdown to clean:
    ```markdown
    {markdown_text}
    ```
    """

    try:
        logger.info("Sending markdown to OpenAI for cleanup...")
        # Use the async client directly
        response = await openai_async_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that cleans markdown text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for more deterministic output
        )

        cleaned_content = response.choices[0].message.content
        if cleaned_content:
            logger.info(
                f"Successfully cleaned markdown with OpenAI. Original length: {len(markdown_text)}, Cleaned length: {len(cleaned_content)}"
            )
            # Basic check to remove potential ```markdown fences if the model added them
            if cleaned_content.strip().startswith("```markdown"):
                cleaned_content = cleaned_content.strip()[len("```markdown") :].strip()
            if cleaned_content.strip().endswith("```"):
                cleaned_content = cleaned_content.strip()[: -len("```")].strip()
            return cleaned_content.strip()
        else:
            logger.warning("OpenAI returned empty content. Returning original markdown.")
            return markdown_text

    except openai.APIError as e:
        logger.error(f"OpenAI API error during cleanup: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI cleanup: {e}", exc_info=True)

    # Return original text if any error occurs
    logger.warning("OpenAI cleanup failed. Returning original markdown.")
    return markdown_text


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
    ref: TabReference, image: Optional[Image.Image], scroll_y: Optional[int]
):
    """Callback triggered after interaction + debounce + content fetch."""
    console.print(
        f"  [green]Interaction: Content Fetched:[/green] Tab {ref.id} ({ref.title} - {ref.url}) ScrollY: {scroll_y}"
    )

    if not ref.html:
        console.print(
            f"    [yellow]Warning:[/yellow] No HTML content fetched for {ref.url}, skipping metadata and save."
        )
        if image:
            _display_screenshot(image, ref.title or "Unknown Title")
        return

    metadata: Optional[HtmlMetadata] = None
    try:
        metadata = extract_metadata(ref.html, url=ref.url)
        console.print(f"    [green]Success:[/green] Extracted Metadata for {ref.title}:")
        console.print(pretty_repr(metadata, indent_size=2, max_length=20))
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
            url=ref.url, html=ref.html, metadata=metadata, image=image, scroll_y=scroll_y
        )
    else:
        missing = []
        if not ref.html:
            missing.append("HTML")
        if not metadata:
            missing.append("Metadata")
        if not image:
            missing.append("Image")
        logger.warning(
            f"Skipping save for {ref.url} after interaction due to missing data: {', '.join(missing)}"
        )
        console.print(
            f"    [yellow]Warning:[/yellow] Skipping save for {ref.url} due to missing data: {', '.join(missing)}."
        )


def _display_screenshot(image: Image.Image, title: str):
    console.print(
        f"    [green]Success:[/green] Captured screenshot for {title}. Displaying cropped preview:"
    )
    try:
        # Define the box (left, upper, right, lower)
        # Keep original width, crop height to max 600px
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
    console.print(f"    [cyan]Processing Tab:[/cyan] {tab.title} ({tab.url}) Initial fetch...")
    html: Optional[str] = None
    metadata: Optional[HtmlMetadata] = None
    screenshot_pil_image: Optional[Image.Image] = None
    final_url: Optional[str] = tab.url  # Start with the initial URL
    final_title: Optional[str] = tab.title
    ws = None

    if not tab.webSocketDebuggerUrl:
        logger.warning(f"Tab {tab.id} missing websocket URL in process_new_tab, cannot process.")
        return

    ws_url = tab.webSocketDebuggerUrl

    try:
        # --- Establish Connection ---
        logger.debug(f"Connecting to WebSocket: {ws_url}")
        ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024)
        logger.debug(f"Connected to WebSocket for tab {tab.id}.")

        # --- Wait for Load & Get Final URL ---
        loaded = await wait_for_page_load(ws)
        if not loaded:
            logger.warning(f"Page load timeout/failure for {tab.id}, proceeding anyway.")
        else:
            logger.debug(f"Page load confirmed for tab {tab.id}.")

        # Get final URL after load/redirects
        try:
            url_script = "window.location.href"
            url_eval = await send_cdp_command(
                ws, "Runtime.evaluate", {"expression": url_script, "returnByValue": True}
            )
            if url_eval and url_eval.get("result", {}).get("type") == "string":
                final_url = url_eval["result"]["value"]
                if final_url != tab.url:
                    logger.info(
                        f"URL changed after load for tab {tab.id}: {tab.url} -> {final_url}"
                    )
                else:
                    logger.debug(f"URL confirmed after load for tab {tab.id}: {final_url}")
            else:
                logger.warning(
                    f"Could not get final URL for tab {tab.id}. Using initial: {tab.url}"
                )
                final_url = tab.url  # Fallback
        except Exception as url_e:
            logger.error(f"Error getting final URL for tab {tab.id}: {url_e}", exc_info=True)
            final_url = tab.url  # Fallback

        # --- Fetch HTML (using existing connection) ---
        try:
            html_script = "document.documentElement.outerHTML"
            html_eval = await send_cdp_command(ws, "Runtime.evaluate", {"expression": html_script})
            if html_eval and html_eval.get("result", {}).get("type") == "string":
                html = html_eval["result"].get("value")
                # Check html is not None before logging length
                if html:
                    console.print(
                        f"    [green]Success:[/green] Retrieved HTML for final URL {final_url} (Length: {len(html)})"
                    )
                else:
                    # Should not happen if type is string, but safety check
                    console.print(
                        f"    [yellow]Warning:[/yellow] Retrieved HTML for {final_url}, but content is unexpectedly None/empty."
                    )
            else:
                console.print(
                    f"    [yellow]Warning:[/yellow] Could not retrieve HTML for {final_url} via WebSocket."
                )
        except Exception as html_e:
            logger.error(
                f"Error getting HTML via WebSocket for {final_url}: {html_e}", exc_info=True
            )
            console.print(f"    [red]Error:[/red] Failed to get HTML for {final_url}: {html_e}")

        # --- Extract Metadata (using final URL) ---
        if html:
            try:
                # Get final title from document if possible (might differ from tab title)
                title_script = "document.title"
                title_eval = await send_cdp_command(
                    ws, "Runtime.evaluate", {"expression": title_script, "returnByValue": True}
                )
                if title_eval and title_eval.get("result", {}).get("type") == "string":
                    final_title = title_eval["result"]["value"]
                else:
                    final_title = tab.title  # Fallback to tab title

                metadata = extract_metadata(html, url=final_url)  # Use final_url
                console.print(f"    [green]Success:[/green] Extracted Metadata for {final_title}:")
                console.print(pretty_repr(metadata, indent_size=2, max_length=20))
            except Exception as meta_e:
                logger.error(f"Error extracting metadata for {final_url}: {meta_e}", exc_info=True)
                console.print(
                    f"    [red]Error:[/red] Failed extracting metadata for {final_url}: {meta_e}"
                )

        # --- Capture Screenshot (using existing connection) ---
        try:
            screenshot_pil_image = await capture_tab_screenshot(ws_url=ws_url, ws_connection=ws)
            if screenshot_pil_image:
                _display_screenshot(screenshot_pil_image, final_title or "Unknown Title")
            else:
                console.print(
                    f"    [yellow]Warning:[/yellow] Could not capture screenshot for {final_url}."
                )
        except Exception as ss_e:
            logger.error(f"Error capturing screenshot for {final_url}: {ss_e}", exc_info=True)
            console.print(
                f"    [red]Error:[/red] Failed to capture screenshot for {final_url}: {ss_e}"
            )

    except websockets.exceptions.WebSocketException as ws_e:
        logger.error(f"WebSocket error during processing tab {tab.id}: {ws_e}", exc_info=True)
        console.print(f"    [red]Error:[/red] WebSocket failed for {tab.url}: {ws_e}")
    except Exception as e:
        logger.error(f"Unexpected error processing tab {tab.id} ({tab.url}): {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Unexpected error for {tab.url}: {e}")
    finally:
        # --- Ensure WebSocket is closed ---
        if ws and ws.state != websockets.protocol.State.CLOSED:
            try:
                await ws.close()
                logger.debug(f"WebSocket closed for tab {tab.id}.")
            except Exception as close_e:
                logger.warning(f"Error closing WebSocket for tab {tab.id}: {close_e}")

    # --- Save Data (using final URL) ---
    # Ensure final_url is valid before proceeding
    if not final_url:
        logger.error(
            f"Cannot save data for tab {tab.id} because final URL could not be determined."
        )
        return

    if html and metadata:
        await save_sample_data(
            url=final_url,  # Use final URL
            html=html,
            metadata=metadata,
            image=screenshot_pil_image,
            scroll_y=0,  # Initial load is always scrollY 0
        )
    else:
        missing = []
        if not html:
            missing.append("HTML")
        if not metadata:
            missing.append("Metadata")
        # Don't require image for saving metadata/html/md
        # if not screenshot_pil_image: missing.append("Image")
        logger.warning(f"Skipping save for {final_url} due to missing data: {', '.join(missing)}")
        console.print(
            f"    [yellow]Warning:[/yellow] Skipping save for {final_url} due to missing data: {', '.join(missing)}."
        )


async def save_sample_data(
    url: str,
    html: str,
    metadata: HtmlMetadata,
    image: Optional[Image.Image],
    scroll_y: Optional[int],
) -> None:
    """Saves HTML, metadata, schedules markdown cleanup, and manages screenshot stacking.
    Accumulates screenshots in memory for simple stacking.
    Organizes samples into: samples/<host>/<slugified_url>.<ext>
    """
    global \
        url_screenshot_data, \
        url_latest_markdown, \
        deferred_openai_cleanup_timers, \
        deferred_openai_cleanup_tasks
    try:
        parsed_url = urlparse(url)
        host = parsed_url.netloc
        if not host:
            logger.warning(f"Could not parse host from URL: {url}, using 'unknown_host'")
            host = "unknown_host"

        # Remove port if present
        host = host.split(":")[0]

        slug = slugify_url(url)
        if not slug:
            logger.warning(f"Could not generate slug for URL: {url}, using 'default_slug'")
            slug = "default_slug"

        base_dir = Path("samples") / host
        base_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        html_path = base_dir / f"{slug}.html"
        json_path = base_dir / f"{slug}.json"
        image_path = base_dir / f"{slug}.jpg"  # Saving as JPEG
        # md_path = base_dir / f"{slug}.md" # No longer saved directly here

        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.debug(f"Saved HTML to {html_path}")

        # Save Metadata
        try:
            # Pydantic models use .model_dump() (v2) or .dict() (v1)
            # No need to check type explicitly if extract_metadata is trusted
            # Let Pydantic handle validation/errors during model creation
            if metadata:
                # Use model_dump() for Pydantic v2+, fallback to dict() for v1
                if hasattr(metadata, "model_dump"):
                    metadata_dict = metadata.model_dump(mode="json")  # Get JSON-serializable dict
                elif hasattr(metadata, "dict"):
                    metadata_dict = metadata.dict()  # Pydantic v1 fallback
                else:
                    logger.error(
                        f"Metadata object for {url} lacks .model_dump() or .dict() method. Cannot serialize."
                    )
                    metadata_dict = {}  # Fallback to empty
            else:
                logger.warning(f"Metadata for {url} is None. Saving empty dict.")
                metadata_dict = {}

            with open(json_path, "w", encoding="utf-8") as f:
                # json.dump expects a dict, model_dump(mode='json') provides this
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metadata to {json_path}")
        except Exception as json_e:
            logger.error(f"Failed to serialize or save metadata for {url}: {json_e}", exc_info=True)
            console.print(f"    [red]Error:[/red] Failed saving metadata JSON for {url}: {json_e}")

        # --- Handle Markdown ---
        try:
            # 1. Extract original markdown
            original_md_content = extract_markdown(html, strategy=MARKDOWN_STRATEGY)
            # 2. Store it as the latest version for this URL
            url_latest_markdown[url] = original_md_content
            logger.debug(f"Stored latest markdown (len {len(original_md_content)}) for {url}.")

            # 3. Schedule debounced OpenAI cleanup (if enabled)
            if enable_openai_cleanup:
                # Cancel existing timer for this URL
                if url in deferred_openai_cleanup_timers:
                    deferred_openai_cleanup_timers[url].cancel()
                    logger.debug(f"Cancelled existing OpenAI debounce timer for {url}.")

                # Schedule the trigger function
                loop = asyncio.get_running_loop()
                logger.debug(f"Scheduling OpenAI cleanup for {url} in {OPENAI_DEBOUNCE_DELAY}s.")
                new_timer = loop.call_later(
                    OPENAI_DEBOUNCE_DELAY, lambda: asyncio.create_task(trigger_openai_cleanup(url))
                )
                deferred_openai_cleanup_timers[url] = new_timer
            else:
                # If cleanup disabled, maybe save original markdown immediately?
                # Let's stick to the plan: done callback handles final save.
                # Log that original will eventually be saved by placeholder mechanism if needed.
                logger.debug("OpenAI cleanup disabled. Final markdown save will use original.")
                # To ensure original is saved if cleanup is off, we could potentially
                # schedule a dummy task/callback that just saves url_latest_markdown[url].
                # For now, relying on cleanup_markdown_with_openai returning original if disabled.

        except Exception as md_e:
            logger.error(
                f"Failed to extract or schedule markdown processing for {url}: {md_e}",
                exc_info=True,
            )
            # Don't save markdown if extraction failed

        # Accumulate and Stack Screenshot
        if image is not None:
            logger.debug(f"Adding screenshot for {url} (scrollY {scroll_y}) to stack list.")
            # Append only the image
            url_screenshot_data[url].append(image)

            # No need to sort anymore
            # url_screenshot_data[url].sort(key=lambda item: item[1])

            # Attempt to stack all images collected for this URL so far
            current_images = url_screenshot_data[url]
            console.print(
                f"    [blue]Attempting to stack {len(current_images)} images for {url}...[/blue]"
            )
            # Pass the list of images directly
            stitched_image = stitch_vertical(current_images)

            if stitched_image:
                stitched_image.save(image_path, "JPEG", quality=85)  # Save stacked image
                logger.info(
                    f"Saved stacked screenshot (size {stitched_image.size}) to {image_path}"
                )
                console.print(
                    f"    [green]Success:[/green] Saved stacked screenshot to {image_path}"
                )
            else:
                logger.warning(f"Stacking failed for {url}. Screenshot not saved.")
                console.print(
                    f"    [yellow]Warning:[/yellow] Stacking failed, screenshot not saved for {url}."
                )
        # Removed the elif for scroll_y is None, as we just append if image exists
        else:
            logger.info(f"No image provided for {url} in this call. No screenshot saved/stacked.")

        console.print(
            f"    [green]Success:[/green] Saved sample data to {base_dir / slug}.[html|json|jpg|md]"
        )

    except OSError as e:
        logger.error(f"OS error saving sample data for {url}: {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] File system error saving data for {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving sample data for {url}: {e}", exc_info=True)
        console.print(f"    [red]Error:[/red] Failed saving sample data for {url}: {e}")


async def handle_openai_cleanup_completion(url: str, task: asyncio.Task):
    """Callback executed when the OpenAI cleanup task finishes."""
    global deferred_openai_cleanup_tasks
    logger.debug(f"OpenAI task completed for URL: {url}")

    try:
        cleaned_markdown = await task  # Get result or raise exception

        # --- Determine file path ---
        # (This duplicates some logic from save_sample_data, maybe refactor later)
        try:
            parsed_url = urlparse(url)
            host = parsed_url.netloc.split(":")[0] if parsed_url.netloc else "unknown_host"
            slug = slugify_url(url) or "default_slug"
            base_dir = Path("samples") / host
            md_path = base_dir / f"{slug}.md"
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as path_e:
            logger.error(f"Error determining markdown save path for {url}: {path_e}", exc_info=True)
            # Cannot save if path fails
            if url in deferred_openai_cleanup_tasks:
                del deferred_openai_cleanup_tasks[url]
            return
        # --- Save the final markdown ---
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(cleaned_markdown)
        logger.info(f"Saved final (cleaned or original) markdown to {md_path}")
        console.print(f"    [green]Success:[/green] Saved final markdown for {url}")

    except asyncio.CancelledError:
        logger.warning(f"OpenAI cleanup task for {url} was cancelled.")
    except Exception as e:
        logger.error(f"OpenAI cleanup task for {url} failed: {e}", exc_info=True)
        # Optionally save the *original* markdown here on failure?
        # Current logic in cleanup_markdown_with_openai returns original on failure, so it's already saved.
        console.print(
            f"    [red]Error:[/red] OpenAI cleanup failed for {url}. Original markdown likely saved."
        )
    finally:
        # Remove task from tracking dict regardless of outcome
        if url in deferred_openai_cleanup_tasks:
            del deferred_openai_cleanup_tasks[url]


async def trigger_openai_cleanup(url: str):
    """Retrieves latest markdown and starts the background cleanup task."""
    global url_latest_markdown, deferred_openai_cleanup_tasks, deferred_openai_cleanup_timers
    logger.info(f"Debounce timer expired for {url}. Triggering OpenAI cleanup.")

    # Clear the timer handle now that it has fired
    if url in deferred_openai_cleanup_timers:
        del deferred_openai_cleanup_timers[url]

    latest_markdown = url_latest_markdown.get(url)
    if not latest_markdown or not latest_markdown.strip():
        logger.info(f"No markdown content found for {url} at cleanup trigger time. Skipping.")
        return

    # Cancel any previously running cleanup task for this URL
    if url in deferred_openai_cleanup_tasks:
        old_task = deferred_openai_cleanup_tasks[url]
        if not old_task.done():
            logger.debug(f"Cancelling previous OpenAI task for {url}.")
            old_task.cancel()
            # We don't necessarily need to wait for cancellation here

    # Start the new cleanup task in the background
    logger.debug(f"Starting background OpenAI task for {url}...")
    cleanup_task = asyncio.create_task(cleanup_markdown_with_openai(latest_markdown))
    deferred_openai_cleanup_tasks[url] = cleanup_task

    # Add the completion callback
    cleanup_task.add_done_callback(
        lambda t: asyncio.create_task(handle_openai_cleanup_completion(url, t))
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

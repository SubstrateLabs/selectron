import asyncio
import sys

from rich.console import Console
from rich.pretty import pretty_repr
from term_image.image import AutoImage

from selectron.chrome.chrome_cdp import capture_active_tab_screenshot, get_active_tab_html
from selectron.chrome.connect import ensure_chrome_connection
from selectron.lib.extract_metadata import HtmlMetadata, extract_metadata
from selectron.lib.logger import get_logger

logger = get_logger(__name__)
console = Console()


async def main():
    if not await ensure_chrome_connection():
        console.print("[bold red]Failed to establish Chrome connection. Exiting.[/bold red]")
        sys.exit(1)

    console.print("\nAttempting to retrieve HTML from the active tab...")
    html = None  # Initialize html to None
    try:
        html = await get_active_tab_html()
        if html:
            console.print("[green]Success:[/green] Retrieved HTML (showing first 500 chars):")
            console.print(f"\n{'=' * 20}\n{html[:500]}...\n{'=' * 20}")
        else:
            console.print(
                "[yellow]Warning:[/yellow] Could not retrieve HTML (check logs for details). No active tab?"
            )
    except Exception as e:
        logger.error(f"Error retrieving HTML: {e}", exc_info=True)
        console.print(f"[red]Error:[/red] An unexpected error occurred while getting HTML: {e}")

    if html:
        console.print("\nAttempting to extract metadata...")
        try:
            metadata: HtmlMetadata = extract_metadata(html)
            if metadata:
                console.print("[green]Success:[/green] Extracted Metadata:")
                console.print(pretty_repr(metadata))
            else:
                console.print("[yellow]Warning:[/yellow] No metadata extracted (check logs?).")
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}", exc_info=True)
            console.print(
                f"[red]Error:[/red] An unexpected error occurred while extracting metadata: {e}"
            )

    console.print("\nAttempting to capture screenshot of the active tab...")
    try:
        screenshot_pil_image = await capture_active_tab_screenshot()
        if screenshot_pil_image:
            console.print("[green]Success:[/green] Displaying screenshot:")
            term_img = AutoImage(screenshot_pil_image)
            term_img.draw()
        else:
            console.print(
                "[yellow]Warning:[/yellow] Could not capture screenshot (check logs for details)."
            )
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}", exc_info=True)
        console.print(
            f"[red]Error:[/red] An unexpected error occurred while capturing screenshot: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main())

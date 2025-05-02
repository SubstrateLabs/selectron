import asyncio

from rich.console import Console
from rich.prompt import Confirm

from selectron.chrome.chrome_launcher import (
    is_chrome_debug_port_active,
    is_chrome_process_running,
    launch_chrome,
    restart_chrome_with_debug_port,
)
from selectron.util.logger import get_logger

logger = get_logger(__name__)
console = Console()


async def ensure_chrome_connection() -> bool:
    """Checks for Chrome debug port and attempts to launch/restart if needed."""
    if await is_chrome_debug_port_active():
        console.print("[green]Success:[/green] Chrome is running with the debug port active.")
        return True

    console.print("[yellow]Warning:[/yellow] Chrome debug port is not accessible.")

    chrome_running = await is_chrome_process_running()

    if not chrome_running:
        console.print("No Chrome processes detected.")
        if Confirm.ask("Do you want to launch Chrome with the debug port enabled?", default=True):
            if await launch_chrome(quiet=True):
                console.print("[green]Success:[/green] Chrome launched with debug port.")
                return True
            else:
                console.print("[red]Error:[/red] Failed to launch Chrome.")
                return False
        else:
            console.print("Exiting without launching Chrome.")
            return False
    else:
        console.print(
            "Chrome process(es) are running, but the debug port is not active or accessible."
        )
        if Confirm.ask(
            "Do you want to quit existing Chrome instances and relaunch with the debug port?",
            default=True,
        ):
            if await restart_chrome_with_debug_port(quiet=True):
                console.print("[green]Success:[/green] Chrome restarted with debug port.")
                return True
            else:
                console.print(
                    "[red]Error:[/red] Failed to restart Chrome. Manual intervention might be required."
                )
                return False
        else:
            console.print("Exiting without restarting Chrome.")
            return False


if __name__ == "__main__":

    async def main_test():
        success = await ensure_chrome_connection()
        if success:
            console.print("Connection check successful.")
        else:
            console.print("Connection check failed or was aborted.")

    asyncio.run(main_test())

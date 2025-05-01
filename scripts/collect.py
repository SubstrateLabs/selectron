import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from selectron.internal.chrome_process import (
    REMOTE_DEBUG_PORT,
    is_chrome_debug_port_active,
    is_chrome_process_running,
    launch_chrome,
    restart_chrome_with_debug_port,
)
from selectron.internal.logger import get_logger

logger = get_logger(__name__)
console = Console()


async def main():
    console.print(
        Panel(
            f"Checking Chrome connection on port {REMOTE_DEBUG_PORT}",
            title="Selectron Chrome Setup",
            expand=False,
        )
    )

    if await is_chrome_debug_port_active():
        console.print("[green]Success:[/green] Chrome is running with the debug port active.")
        return

    console.print("[yellow]Warning:[/yellow] Chrome debug port is not accessible.")

    chrome_running = await is_chrome_process_running()

    if not chrome_running:
        console.print("No Chrome processes detected.")
        if Confirm.ask("Do you want to launch Chrome with the debug port enabled?", default=True):
            console.print("Attempting to launch Chrome...")
            if await launch_chrome(quiet=True):
                console.print("[green]Success:[/green] Chrome launched with debug port.")
            else:
                console.print("[red]Error:[/red] Failed to launch Chrome.")
        else:
            console.print("Exiting without launching Chrome.")
    else:
        console.print(
            "Chrome process(es) are running, but the debug port is not active or accessible."
        )
        if Confirm.ask(
            "Do you want to quit existing Chrome instances and relaunch with the debug port?",
            default=True,
        ):
            console.print("Attempting to restart Chrome...")
            if await restart_chrome_with_debug_port(quiet=True):
                console.print("[green]Success:[/green] Chrome restarted with debug port.")
            else:
                console.print(
                    "[red]Error:[/red] Failed to restart Chrome. Manual intervention might be required."
                )
        else:
            console.print("Exiting without restarting Chrome.")


if __name__ == "__main__":
    asyncio.run(main())

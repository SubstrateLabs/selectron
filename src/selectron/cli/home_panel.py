from typing import Literal

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

ChromeStatus = Literal[
    "unknown",
    "checking",
    "not_running",
    "no_debug_port",
    "ready_to_connect",
    "connecting",
    "connected",
    "error",
]


class HomePanel(Container):
    """A panel to display connection status and offer connection initiation."""

    status: reactive[ChromeStatus] = reactive[ChromeStatus]("unknown")

    DEFAULT_CSS = """
    HomePanel {
        align: center top;
    }
    #home-status-content {
        margin-bottom: 1;
    }
    #agent-status-label {
        margin-top: 1;
        text-align: center;
        width: 100%;
    }
    Button {
        margin-top: 1;
        min-width: 30; # Make buttons wider
    }
    """

    def compose(self) -> ComposeResult:
        # This container will hold the dynamically changing content
        yield Vertical(
            Vertical(id="home-status-content"),  # Container for status widgets
            Label("Interact with a page in Chrome to get started", id="agent-status-label"),  # Agent status label always present below
        )

    def watch_status(self, old_status: ChromeStatus, new_status: ChromeStatus) -> None:
        """Update the UI when the status reactive changes."""
        self.update_ui(new_status)

    def on_mount(self) -> None:
        """Initial UI setup based on the initial status."""
        self.update_ui(self.status)

    def update_ui(self, status: ChromeStatus) -> None:
        status_container = self.query_one(
            "#home-status-content", Vertical
        )  # Target the inner container

        async def clear_and_mount():
            await status_container.remove_children()  # Clear the inner container
            widgets_to_mount = []
            if status == "unknown":
                widgets_to_mount = [Static("Checking Chrome status...")]
            elif status == "checking":
                widgets_to_mount = [Static("Checking Chrome status...")]
            elif status == "not_running":
                widgets_to_mount = [
                    Static("Chrome is not running."),
                    Button("Launch Chrome", id="launch-chrome", variant="primary"),
                ]
            elif status == "no_debug_port":
                widgets_to_mount = [
                    Static("Chrome is running, but the debug port is not active."),
                    Button(
                        "Restart Chrome with Debug Port", id="restart-chrome", variant="primary"
                    ),
                ]
            elif status == "ready_to_connect":
                widgets_to_mount = [
                    Static("Chrome ready. Connecting monitor..."),
                ]
            elif status == "connecting":
                widgets_to_mount = [Static("Connecting monitor...")]
            elif status == "connected":
                widgets_to_mount = [
                    Static("Connected to Chrome.", classes="success-message"),
                ]
            elif status == "error":
                widgets_to_mount = [
                    Static("Error connecting or checking Chrome status.", classes="error-message"),
                    Button("Retry Status Check", id="check-chrome-status", variant="error"),
                ]

            # Mount the new widgets
            for widget in widgets_to_mount:
                await status_container.mount(widget)  # Mount into the inner container

        self.app.call_later(clear_and_mount)

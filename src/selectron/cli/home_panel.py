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

AiStatus = Literal["enabled_anthropic", "enabled_openai", "disabled"]


class HomePanel(Container):
    """A panel to display connection status and offer connection initiation."""

    chrome_status: reactive[ChromeStatus] = reactive[ChromeStatus]("unknown")
    ai_status: reactive[AiStatus] = reactive[AiStatus]("disabled")  # Default to disabled

    def compose(self) -> ComposeResult:
        # This container will hold the dynamically changing content
        yield Vertical(
            Vertical(id="home-status-content"),  # Container for chrome status widgets
            Label("AI status: checking...", id="ai-status-label"),  # AI status label
            Label(
                "Interact with a page in Chrome to get started", id="agent-status-label"
            ),  # Agent status label always present below
        )

    def watch_chrome_status(self, old_status: ChromeStatus, new_status: ChromeStatus) -> None:
        """Update the UI when the chrome_status reactive changes."""
        self.update_chrome_ui(new_status)

    def watch_ai_status(self, old_ai_status: AiStatus, new_ai_status: AiStatus) -> None:
        """Update the AI status label when the ai_status reactive changes."""
        self.update_ai_ui(new_ai_status)

    def on_mount(self) -> None:
        """Initial UI setup based on the initial status."""
        self.update_chrome_ui(self.chrome_status)
        self.update_ai_ui(self.ai_status)  # Initial AI status update

    def update_ai_ui(self, status: AiStatus) -> None:
        """Updates the dedicated AI status label."""
        ai_label = self.query_one("#ai-status-label", Label)
        message = "AI status: Unknown"
        css_class = ""
        if status == "enabled_anthropic":
            message = "AI Enabled (Anthropic)"
            css_class = "success-message"
        elif status == "enabled_openai":
            message = "AI Enabled (OpenAI)"
            css_class = "success-message"
        elif status == "disabled":
            message = "AI Disabled (No API key found)"
            css_class = "warning-message"

        ai_label.update(message)
        ai_label.set_classes(css_class)  # Apply styling class

    def update_chrome_ui(self, status: ChromeStatus) -> None:
        status_container = self.query_one(
            "#home-status-content", Vertical
        )  # Target the inner container

        async def clear_and_mount():
            await status_container.remove_children()  # Clear the inner container
            widgets_to_mount = []
            # Use class attributes for status messages for styling
            if status == "unknown":
                widgets_to_mount = [Static("Checking Chrome status...")]
            elif status == "checking":
                widgets_to_mount = [Static("Checking Chrome status...")]
            elif status == "not_running":
                widgets_to_mount = [
                    Static("Chrome: Not running.", classes="warning-message"),
                    Button("Launch Chrome", id="launch-chrome", variant="warning"),
                ]
            elif status == "no_debug_port":
                widgets_to_mount = [
                    Static("Chrome: Running, Debug port inactive.", classes="warning-message"),
                    Button("Restart Chrome w/ Debug Port", id="restart-chrome", variant="warning"),
                ]
            elif status == "ready_to_connect":
                widgets_to_mount = [
                    Static("Chrome: Ready to connect.", classes="success-message"),
                ]
            elif status == "connecting":
                widgets_to_mount = [Static("Chrome: Connecting monitor...")]
            elif status == "connected":
                widgets_to_mount = [
                    Static("Chrome: Connected.", classes="success-message"),
                ]
            elif status == "error":
                widgets_to_mount = [
                    Static("Chrome: Error checking status.", classes="error-message"),
                    Button("Retry Status Check", id="check-chrome-status", variant="error"),
                ]

            # Mount the new widgets
            for widget in widgets_to_mount:
                await status_container.mount(widget)  # Mount into the inner container

        self.app.call_later(clear_and_mount)

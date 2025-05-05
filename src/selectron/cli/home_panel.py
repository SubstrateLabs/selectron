import logging
from typing import Literal

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

logger = logging.getLogger(__name__)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial values for reactive attributes
        self.chrome_status = "checking"
        self.ai_status = "disabled"

        # Define buttons in __init__
        self.launch_chrome_button = Button(
            "🚀 Launch Chrome", id="launch-chrome", variant="primary"
        )
        self.restart_chrome_button = Button(
            "🔄 Restart Chrome (with debug)", id="restart-chrome", variant="warning"
        )
        self.open_duckdb_button = Button("🦆 Open DuckDB UI", id="open-duckdb", variant="default")

    def compose(self) -> ComposeResult:
        yield Vertical(
            # Chrome Status Section
            Label("Chrome Connection Status", classes="section-title"),
            Vertical(id="home-status-content"),  # Dynamic content here
            # AI Status Section (moved up)
            Label("AI Status", classes="section-title ai-title"),
            Static(id="ai-status-text", classes="status-text"),
            # Agent Status Section (moved up)
            Label("Agent Status", classes="section-title"),
            Static("Interact with a page in Chrome to get started", id="agent-status-label"),
            # Utility Buttons Section (at the bottom)
            Horizontal(
                self.open_duckdb_button,
                classes="button-group utility-buttons",
            ),
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

        async def _update_label():
            try:
                ai_widget = self.query_one("#ai-status-text", Static)
            except NoMatches:
                logger.error("Failed to find #ai-status-text container during update.")
                return

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

            ai_widget.update(message)
            ai_widget.set_classes(css_class)  # Apply styling class

        self.app.call_later(_update_label)

    def update_chrome_ui(self, status: ChromeStatus) -> None:
        async def clear_and_mount():
            try:
                status_container = self.query_one(
                    "#home-status-content", Vertical
                )  # Target the inner container
            except NoMatches:
                logger.error("Failed to find #home-status-content container during update.")
                return

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

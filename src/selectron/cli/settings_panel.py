from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Static


class SettingsPanel(VerticalScroll):
    """Container for application settings and potentially destructive actions."""

    DEFAULT_CSS = """
    SettingsPanel {
        padding: 1 2; /* Add padding around the content */
    }
    SettingsPanel > Vertical {
         /* Add space between button and potential future elements */
        grid-size: 2;
        grid-gutter: 1 2;
        padding: 1 0;
    }
    #drop-tables-section {
        border: round red;
        padding: 1 2;
        margin-bottom: 1;
    }

    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="drop-tables-section"):
            yield Static("[bold red]Danger Zone[/]", classes="header")
            yield Static("This will permanently delete all data saved by Selectron.")
            yield Button(
                "Drop All DuckDB Tables",
                id="drop-all-tables",
                variant="error",
                classes="settings-button",
            )
        # Add placeholders for future settings sections if needed
        # yield Static("Future settings section...")

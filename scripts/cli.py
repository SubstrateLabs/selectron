from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header


class CliApp(App[None]):
    """A basic Textual app."""

    BINDINGS = [
        Binding(
            key="ctrl+c",
            action="quit",
            description="Quit App",
            show=True,
        ),
        Binding(
            key="ctrl+q",
            action="quit",
            description="Quit App",
            show=True,
        ),
    ]

    # Initialize dark mode state
    dark = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Container(
            # Add your widgets here
        )
        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    # No need to override action_quit, the default one calls self.exit()
    # def action_quit(self) -> None:
    #     """An action to quit the app."""
    #     self.exit()


if __name__ == "__main__":
    app = CliApp()
    app.run()

import sys

from rich import print as rich_print

from selectron.util.model_config import ModelConfig


def start():
    try:
        model_config = ModelConfig()
    except ValueError as e:
        rich_print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    from selectron.cli.app import SelectronApp

    app = SelectronApp(model_config=model_config)
    app.run()


if __name__ == "__main__":
    start()

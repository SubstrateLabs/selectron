import os
import sys

from rich import print as rich_print


def start():
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        rich_print(
            "[bold red]Error:[/bold red] Env missing one of: [cyan]OPENAI_API_KEY[/cyan] or [cyan]ANTHROPIC_API_KEY[/cyan] or [cyan]OPENROUTER_API_KEY[/cyan]"
        )
        sys.exit(1)

    from selectron.cli.app import SelectronApp

    app = SelectronApp(openai_key=openai_key, anthropic_key=anthropic_key)
    app.run()


if __name__ == "__main__":
    start()

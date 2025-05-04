import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic_ai import UnexpectedModelBehavior, capture_run_messages
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from selectron.ai.codegen_agent import CodegenAgent
from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig
from selectron.util.resolve_urls import resolve_urls

logger = get_logger(__name__)


async def main() -> None:
    console = Console()

    # input_file = Path("scripts/news.ycombinator.com.json")
    input_file = Path("scripts/x.com~~2fhome.json")
    if not input_file.exists():
        console.print(f"[red]error: input file not found at {input_file}[/red]")
        return

    try:
        data = json.loads(input_file.read_text())
        raw_samples = data.get("html_elements")
        base_url = data.get("url")

        if not isinstance(raw_samples, list) or not raw_samples:
            console.print(f"[red]error: no 'html_elements' list found in {input_file}[/red]")
            return
        if not base_url or not isinstance(base_url, str):
            console.print(f"[red]error: missing or invalid 'url' (base_url) in {input_file}[/red]")
            return
    except json.JSONDecodeError:
        console.print(f"[red]error: could not parse json file {input_file}[/red]")
        return

    processed_samples = [resolve_urls(html, base_url) for html in raw_samples]
    if not processed_samples:
        console.print("[red]no html samples loaded from file[/red]")
        return

    try:
        # Reinstate capture_run_messages wrapper AND variable capture
        with capture_run_messages() as messages:
            model_config = ModelConfig()
            codegen_agent = CodegenAgent(
                html_samples=processed_samples,
                model_cfg=model_config,
                save_results=True,
                output_dir=Path("src/selectron/parsers"),
                base_url=base_url,
                input_selector=data.get("selector"),
                input_selector_description=data.get("selector_description"),
            )
            code, extracted = await codegen_agent.run()
    except UnexpectedModelBehavior as e:
        console.print(f"[bold red]Agent Error (Retries Likely Exceeded):[/bold red] {e}")
        # Print messages when this specific error occurs
        console.rule("Captured Agent Messages (on failure)")
        console.print(messages)
        console.rule()
        return  # Exit after handling
    except Exception as e:
        console.print(f"[bold red]failed (General Exception):[/bold red] {e}")
        return

    console.rule("generated code")
    syntax = Syntax(code, "python", theme="monokai", word_wrap=True, line_numbers=True)
    console.print(syntax)

    # build table
    console.rule("extracted data")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("sample #", style="dim")

    if extracted:
        all_keys = set().union(*extracted)
        for k in sorted(all_keys):
            table.add_column(k)

        def _repr_val(v: Any) -> str:
            if isinstance(v, (str, int, float)):
                return str(v)
            return json.dumps(v, ensure_ascii=False)

        for idx, d in enumerate(extracted):
            row = [str(idx)] + [_repr_val(d.get(k, "")) for k in sorted(all_keys)]
            table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"uncaught error: {e}")

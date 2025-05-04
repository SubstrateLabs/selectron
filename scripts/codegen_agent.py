# TODO: prototype codegen agent
# writes + evaluates python code that uses bs4 (and no other libraries) (already installed)
# code takes in html for an element on a page and extracts the key information exhaustively
# code return value must be a json object with concise descriptive field names
# the json object must always be str:str (simple flat all-string key-value pairs)
# feedback must be provided to the agent, e.g. any execution or syntax errors in the code,
# as well as verifying that the code runs on the input html and produces non-empty JSON objects
# matching the above requirements.
# agent is given several examples of the html at the start
# when done, script prints the output code using rich with syntax highlighting (installed)
# then prints the extracted output from all the html elements passed as input, in a table

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig
from selectron.util.resolve_urls import resolve_urls
from selectron.util.sample_items import sample_items

logger = get_logger(__name__)

# =============================================================
# helper fns
# =============================================================


def _validate_result(obj: Any) -> Tuple[bool, str]:
    """Ensure obj is a non-empty dict with string keys and json-serialisable values.

    Allowed value types:
    - str
    - list[Any] (recursively validated so elements are allowed types)
    - dict[str, str]
    """

    def _is_valid_val(val: Any) -> bool:
        if isinstance(val, str):
            return True
        if isinstance(val, dict):
            return all(isinstance(k, str) and isinstance(v, str) for k, v in val.items())
        if isinstance(val, list):
            return all(_is_valid_val(it) for it in val)
        return False

    if not isinstance(obj, dict):
        return False, "result is not a dict"
    if not obj:
        return False, "dict is empty"
    for k, v in obj.items():
        if not isinstance(k, str):
            return False, f"non-string key detected: {k!r}"
        if not _is_valid_val(v):
            return False, f"invalid value for key {k}: {type(v)}"
    return True, "ok"


def _exec_candidate(code: str, html_samples: List[str]) -> Tuple[bool, str, List[Dict[str, str]]]:
    """execute candidate code, run `parse_element` on each sample.

    returns (success, feedback, outputs)
    """
    sandbox: Dict[str, Any] = {
        "BeautifulSoup": BeautifulSoup,
        "json": json,
    }

    try:
        # compile first for clearer syntax errors
        compiled = compile(code, "<agent_code>", "exec")
    except SyntaxError as e:
        feedback = f"syntax error: {e.msg} (line {e.lineno})"
        logger.warning(feedback)
        return False, feedback, []

    try:
        exec(compiled, sandbox)  # noqa: S102 – need exec for dynamic agent code
    except Exception as e:  # granular unknown – return traceback str
        feedback = f"runtime error during exec: {type(e).__name__}: {e}"
        logger.warning(feedback, exc_info=True)
        return False, feedback, []

    parse_fn = sandbox.get("parse_element")
    if not callable(parse_fn):
        feedback = "function `parse_element(html: str) -> dict` not found"
        logger.warning(feedback)
        return False, feedback, []

    outputs: List[Dict[str, str]] = []
    for idx, html in enumerate(html_samples):
        try:
            result = parse_fn(html)  # type: ignore[misc]
        except Exception as e:
            feedback = f"error when calling parse_element on sample {idx}: {type(e).__name__}: {e}"
            logger.warning(feedback, exc_info=True)
            return False, feedback, []

        ok, msg = _validate_result(result)
        if not ok:
            feedback = f"invalid return value for sample {idx}: {msg}"
            logger.warning(feedback)
            return False, feedback, []
        outputs.append(result)  # type: ignore[arg-type]

    return True, "success", outputs


# Pydantic model for the evaluation tool's result
class CodeEvaluationResult(BaseModel):
    success: bool = Field(
        ..., description="Whether the code executed successfully and passed validation."
    )
    feedback: str = Field(
        ..., description="Error message if success is false, or 'success' if true."
    )
    sampled_output: Optional[str] = Field(
        default=None, description="JSON string of sampled outputs if successful, otherwise null."
    )


async def generate_and_test_agent(
    processed_html_samples: List[str], model: str = "gpt-4o-mini"
) -> Tuple[str, List[Dict[str, Any]]]:
    """main loop. returns final code string and extracted outputs."""

    system_prompt = textwrap.dedent(
        """
        context:
        we are building a GENERAL-PURPOSE web-scraping library. the goal is to extract *human-useful* information from **a single element** of html.

        typical things humans care about include (but aren't limited to):
          • primary text content or headline (call it `title`)
          • a **primary** url (e.g. the first / most prominent <a> tag) plus **all** other urls.
          • a stable identifier (`id` attribute) when present.
          • rank / index numbers if present as plain text (e.g. "1." at the start of list rows).
          • images → src, alt, title, width/height, plus any `data-*` attrs.
          • assorted metadata attributes like aria-label, title, data-* etc.

        sometimes an element contains *multiple* links or images. therefore our return schema must allow **lists**.

        TASK:
        1. generate minimal, robust python (only stdlib + BeautifulSoup) that defines ONE function:
             `parse_element(html: str) -> dict[str, str|list|dict]`
           – recommended keys (use when applicable):
             id, title, primary_url, urls (list), images (list[dict]), rank, text, attributes.
           – keys must be concise snake_case.
           – values may be str, list[dict|str], or dict[str,str].
           – the dict MUST be NON-EMPTY on given examples and should capture **as much of the above as is obviously available**.
           – IMPORTANT: **omit keys entirely** if no corresponding data is found (e.g. don't include `images: []`).
        2. never raise inside `parse_element`; fail gracefully.
        3. do NOT perform I/O, prints, or network calls. safe on import.
        4. import `BeautifulSoup` and `re` exactly once at the top if needed.

        TOOL USAGE:
        - use the `evaluate_and_sample_code` tool to test your generated code.
        - the tool will return a `CodeEvaluationResult` object.
        - if `success` is false, read the `feedback` and fix your code.
        - if `success` is true, examine the `sampled_output` (a json string of sample results).
        - **iterate** by calling the tool again with refined code until the `sampled_output` looks correct and contains the desired data fields (id, title, urls, etc.).
        - your FINAL response MUST be the validated python code string itself, and nothing else.
        """
    ).strip()

    # define the tool function within this scope to capture processed_html_samples
    async def evaluate_and_sample_code(code: str) -> CodeEvaluationResult:
        """Evaluates the python code, runs it on samples, and returns feedback with sampled results."""
        # strip surrounding fences if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        elif code.startswith("```"):
            code = code[len("```") :].strip()
        if code.endswith("```"):
            code = code[: -len("```")].strip()

        success, feedback, outputs = _exec_candidate(code, processed_html_samples)

        sampled_output_str: Optional[str] = None
        if success:
            logger.info("agent code passed validation")
            sampled_outputs = sample_items(outputs, sample_size=4)

            sampled_output_str = json.dumps(sampled_outputs, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"agent code failed validation: {feedback}")

        return CodeEvaluationResult(
            success=success, feedback=feedback, sampled_output=sampled_output_str
        )

    # wrap the function as a pydantic-ai tool
    evaluation_tool = Tool(evaluate_and_sample_code)

    agent = Agent(model, tools=[evaluation_tool], system_prompt=system_prompt)

    logger.info("starting agent run with tool-based iteration...")
    response = await agent.run("generate the initial python code and evaluate it using the tool.")
    final_code = str(response.output)
    logger.info("agent run finished.")

    # run the final code one last time to get definitive outputs
    success, feedback, final_outputs = _exec_candidate(final_code, processed_html_samples)
    if not success:
        logger.error(f"final code returned by agent failed validation: {feedback}")
        raise RuntimeError(f"agent returned non-working code: {feedback}")

    return final_code, final_outputs  # type: ignore - validation ensures serializability


async def main() -> None:
    """main async execution function."""
    console = Console()

    input_file = Path("scripts/news.ycombinator.com.json")
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

    # preprocess samples
    processed_samples = [resolve_urls(html, base_url) for html in raw_samples]

    if not processed_samples:
        console.print("[red]no html samples loaded from file[/red]")
        return

    try:
        model_config = ModelConfig()  # raise if key missing
        code, extracted = await generate_and_test_agent(
            processed_samples, model_config.selector_agent_model
        )
    except Exception as e:
        console.print(f"[bold red]failed:[/bold red] {e}")
        return

    # pretty-print code
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

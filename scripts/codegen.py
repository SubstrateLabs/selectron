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

from selectron.ai.codegen_utils import (
    validate_cross_key_duplicates,
    validate_empty_columns,
    validate_identical_columns,
    validate_redundant_key_pairs,
    validate_text_representation,
)
from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig
from selectron.util.resolve_urls import resolve_urls
from selectron.util.sample_items import sample_items

logger = get_logger(__name__)


def _validate_result(obj: Any) -> Tuple[bool, str]:
    """Ensure obj is a non-empty dict with string keys and allowed value types.

    Allowed value types:
    - str
    - int
    - list[Any] (recursively validated so elements are allowed types)
    - dict[str, str]
    """

    def _is_valid_val(val: Any) -> bool:
        if isinstance(val, str):
            return True
        if isinstance(val, int):
            return True
        if isinstance(val, dict):
            # Check for dict[str, str]
            return all(isinstance(k, str) and isinstance(v, str) for k, v in val.items())
        if isinstance(val, list):
            # Explicitly check for list[str | dict[str, str]]
            for item in val:
                is_item_str = isinstance(item, str)
                is_item_dict_str_str = isinstance(item, dict) and all(
                    isinstance(k, str) and isinstance(v, str) for k, v in item.items()
                )
                if not (is_item_str or is_item_dict_str_str):
                    return False  # Item is neither str nor dict[str, str]
            return True  # All items were valid str or dict[str, str]
        return False

    if not isinstance(obj, dict):
        return False, "result is not a dict"
    if not obj:
        return False, "dict is empty"
    for k, v in obj.items():
        if not isinstance(k, str):
            return False, f"non-string key detected: {k!r}"
        if not _is_valid_val(v):
            # If validation fails and the value is a list, find the invalid item
            if isinstance(v, list):
                for idx, item in enumerate(v):
                    is_item_str = isinstance(item, str)
                    is_item_dict_str_str = isinstance(item, dict) and all(
                        isinstance(k, str) and isinstance(v_inner, str)
                        for k, v_inner in item.items()
                    )
                    if not (is_item_str or is_item_dict_str_str):
                        # Construct specific feedback for list item failure
                        return (
                            False,
                            f"invalid item at index {idx} for key '{k}'. Expected str or dict[str, str], but got {type(item)}.",
                        )
                # Should not be reached if _is_valid_val returned False, but as a fallback:
                return (
                    False,
                    f"invalid list value for key '{k}' (reason unclear, check list items).",
                )
            else:
                # Generic feedback for non-list types
                return False, f"invalid value for key '{k}': type {type(v)}"
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
    html_samples: List[str], model_config: ModelConfig
) -> Tuple[str, List[Dict[str, Any]]]:
    """main loop. returns final code string and extracted outputs."""

    system_prompt = textwrap.dedent(
        """
        CONTEXT:
        you are an expert web-scraping developer. your goal is to extract ALL useful information from a single element of html from a webpage.

        TASK:
        1. generate minimal, robust python (only stdlib + BeautifulSoup) that defines ONE function:
             `parse_element(html: str) -> dict[str, str|int|list|dict]`
           – keys must be concise snake_case.
           – values may be str, int, list[dict|str], or dict[str,str].
           – the dict MUST be NON-EMPTY on given examples and should capture **as much of the DISTINCT, USEFUL information exhaustively** WITHOUT introducing duplication.
        2. never raise inside `parse_element`; fail gracefully.
        3. do NOT perform I/O, prints, or network calls. safe on import.
        4. import `BeautifulSoup` and `re` exactly once at the top if needed.

        Start by identifying the values to extract based on the provided HTML examples. Below are some general keys you should always look for.
        However, you should ALWAYS supplement these with additional keys to EXHAUSTIVELY capture all useful information from the elements.
        IMPORTANT: omit keys entirely if no corresponding data is found. Aim for Mutually Exclusive, Collectively Exhaustive (MECE) results – avoid storing the same piece of data under multiple keys.

        GENERAL KEYS TO CONSIDER (adapt and add based on specific html):
        - **URLs**:
            - `primary_url`: The most prominent link (often the permalink/status link).
            - `urls`: A list of ALL *other* distinct URLs found (EXCLUDE the `primary_url` from this list).
        - **Identification**:
            - `id`: A stable identifier for the element (e.g., from a `data-id` attribute or part of a permalink).
            - `title`: A primary title or heading (look in `h*`, `title` tags, `aria-label`).
        - **Author Information**:
            - `author`: The display name of the author.
            - `author_url`: The URL to the author's profile.
            - `author_handle`: The author's handle (often starts with '@', look near the author name/url).
            - `author_avatar_url`: URL of the author's avatar/profile image (often within the `images` list, but can be separate if distinct).
        - **Content**:
            - `description`: The main text content (look in `<p>`, `div[data-testid='tweetText']`, etc.; fall back to combined visible text if specific tags fail).
            - `images`: A list of dictionaries for each image, containing `src`, `alt`, `title`, `width`, `height`, and relevant `data-*` attributes. EXCLUDE the author avatar if `author_avatar_url` is already populated.
        - **Timestamps**:
            - `timestamp`: Human-readable time (e.g., "17h", "May 3"). Look for `<time>` tags or text near author info.
            - `datetime`: Machine-readable timestamp (e.g., ISO 8601 format). Look for the `datetime` attribute on `<time>` tags.
        - **Metrics/Stats** (look for numbers associated with icons or specific `data-testid` attributes like 'reply', 'retweet', 'like', 'view', 'bookmark'):
            - `reply_count`, `repost_count`, `like_count`, `bookmark_count`, `view_count`: Parse numerical values (handle 'K'/'M' suffixes if present, converting to integers).
        - **Ranking/Position**:
            - `rank`: Ordinal position if applicable (e.g., in search results). Look for `data-rank` or similar attributes.

        TOOL USAGE:
        - use the `evaluate_and_sample_code` tool to test your generated code.
        - the tool will return a `CodeEvaluationResult` object.
        - if `success` is false, read the `feedback` and fix your code.
        - if `success` is true, examine the `sampled_output` (a json string of sample results).
            - IMPORTANT: pay attention to the `feedback` field even when `success` is true. It may contain notes about quality issues (like missing data or redundant fields).
        - you MUST iterate by calling the tool again with refined code until the `sampled_output` is: 
            - fixes any `feedback` returned by the tool
            - correct, exhaustive, and robust, following the TASK guidelines and CONTEXT above.
            - NOTE: the final result is PRODUCTION CODE and must be rigorously reviewed and edited for quality before it is automatically deployed.
        - **ITERATE** based on this quality feedback (e.g., redundant keys identified) AND your own analysis of the `sampled_output` compared to the guidelines.
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

        success, feedback, outputs = _exec_candidate(code, html_samples)

        sampled_output_str: Optional[str] = None
        if success:
            logger.info("agent code passed validation")

            quality_feedback: List[str] = []
            # --- Generic Quality Analysis on *all* outputs ---
            if len(outputs) > 1:
                all_keys = set().union(*(d.keys() for d in outputs))

                # Perform validation using utility functions
                quality_feedback.extend(validate_empty_columns(outputs, all_keys))
                quality_feedback.extend(validate_identical_columns(outputs, all_keys))
                quality_feedback.extend(validate_text_representation(outputs, html_samples))
                quality_feedback.extend(validate_redundant_key_pairs(outputs, all_keys))
                quality_feedback.extend(validate_cross_key_duplicates(outputs, all_keys))

            # Prepend quality feedback to execution feedback
            if quality_feedback:
                feedback = (
                    "Quality issues detected:\n- "
                    + "\n- ".join(quality_feedback)
                    + "\n\nOriginal status: "
                    + feedback
                )
            # --- End Quality Analysis ---

            # Get sampled outputs with original indices
            sampled_outputs_with_indices = sample_items(outputs, sample_size=4)

            sampled_outputs_for_json: List[Dict[str, Any]] = []

            for _, sample_dict in sampled_outputs_with_indices:
                sampled_outputs_for_json.append(sample_dict)  # Keep for JSON output

            # Dump only the dictionaries to JSON
            sampled_output_str = json.dumps(sampled_outputs_for_json, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"agent code failed validation: {feedback}")

        return CodeEvaluationResult(
            success=success, feedback=feedback, sampled_output=sampled_output_str
        )

    evaluation_tool = Tool(evaluate_and_sample_code)
    agent = Agent(model_config.agent_model, tools=[evaluation_tool], system_prompt=system_prompt)
    logger.info("starting agent...")
    response = await agent.run("generate the initial python code and evaluate it using the tool.")
    final_code = str(response.output)
    logger.info("agent run finished.")
    # run the final code one last time to get definitive outputs
    success, feedback, final_outputs = _exec_candidate(final_code, html_samples)
    if not success:
        logger.error(f"final code returned by agent failed validation: {feedback}")
        raise RuntimeError(f"agent returned non-working code: {feedback}")

    return final_code, final_outputs  # type: ignore - validation ensures serializability


async def main() -> None:
    """main async execution function."""
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
        model_config = ModelConfig()
        code, extracted = await generate_and_test_agent(processed_samples, model_config)
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

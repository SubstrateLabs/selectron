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
from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior, capture_run_messages
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from selectron.ai.codegen_utils import (
    validate_cross_key_duplicates,
    validate_empty_columns,
    validate_identical_columns,
    validate_internal_repetition,
    validate_naive_text_match,
    validate_redundant_key_pairs,
    validate_text_representation,
)
from selectron.util.model_config import ModelConfig
from selectron.util.resolve_urls import resolve_urls
from selectron.util.sample_items import sample_items


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
            # Explicitly check for list[str | dict[str, str | int | None]]
            for item in val:
                is_item_str = isinstance(item, str)
                # Check if item is a dict where keys are str and values are str, int, or None
                is_item_dict_flexible_values = isinstance(item, dict) and all(
                    isinstance(k, str) and isinstance(v, (str, int)) or v is None
                    for k, v in item.items()
                )
                if not (is_item_str or is_item_dict_flexible_values):
                    return False  # Item is neither str nor dict[str, str | int | None]
            return True  # All items were valid str or dict[str, str | int | None]
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
                    # Check if item is a dict where keys are str and values are str, int, or None
                    is_item_dict_flexible_values = isinstance(item, dict) and all(
                        isinstance(k, str) and isinstance(v_inner, (str, int)) or v_inner is None
                        for k, v_inner in item.items()
                    )
                    if not (is_item_str or is_item_dict_flexible_values):
                        # Construct specific feedback for list item failure
                        return (
                            False,
                            f"invalid item at index {idx} for key '{k}'. Expected str or dict[str, str | int | None], but got {type(item)} with invalid internal types.",
                        )
                # Should not be reached if _is_valid_val returned False, but as a fallback:
                return (
                    False,
                    f"invalid list value for key '{k}' (reason unclear, check list items).",
                )
            else:
                # Generic feedback for non-list types
                if v is None:
                    return (
                        False,
                        f"invalid value for key '{k}': assigned None. Omit the key entirely if no valid value is found.",
                    )
                else:
                    return False, f"invalid value for key '{k}': type {type(v)}"
    return True, "ok"


async def generate_and_test_agent(
    html_samples: List[str], model_config: ModelConfig
) -> Tuple[str, List[Dict[str, Any]]]:
    """main loop. returns final code string and extracted outputs."""

    console = Console()

    # --- BEGIN CHANGE: Define CodeEvaluationResult inside --- #
    class CodeEvaluationResult(BaseModel):
        success: bool = Field(
            ..., description="Whether the code executed successfully and passed validation."
        )
        feedback: str = Field(
            ...,
            description="Error message if success is false, or 'success' if true (may include quality feedback).",
        )
        sampled_output_with_html: Optional[str] = Field(
            default=None,
            description="JSON string of ONE sampled input/output pair: {'html_input': str, 'extracted_data': dict} if successful, otherwise null.",
        )
        iteration_count: int = Field(
            ..., description="The current evaluation iteration number (starting from 1)."
        )

    # --- END CHANGE --- #

    # --- BEGIN CHANGE: Define _exec_candidate inside --- #
    def _exec_candidate(
        code: str, html_samples_inner: List[str]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """execute candidate code, run `parse_element` on each sample."""
        sandbox: Dict[str, Any] = {"BeautifulSoup": BeautifulSoup, "json": json}
        try:
            compiled = compile(code, "<agent_code>", "exec")
        except SyntaxError as e:
            feedback = f"syntax error: {e.msg} (line {e.lineno})"
            console.print(f"[WARNING] {feedback}")
            try:
                console.rule("[bold red]Syntax Error Detected[/bold red]")
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True, word_wrap=True)
                console.print(syntax)
                console.rule()
            except Exception as print_err:
                console.print(
                    f"[ERROR] Failed to print syntax error details: {print_err}", style="bold red"
                )
            return False, feedback, []
        try:
            exec(compiled, sandbox)
        except Exception as e:
            feedback = f"runtime error during exec: {type(e).__name__}: {e}"
            console.print(f"[WARNING] {feedback}")
            console.print_exception(show_locals=False)
            return False, feedback, []
        parse_fn = sandbox.get("parse_element")
        if not callable(parse_fn):
            feedback = "function `parse_element(html: str) -> dict` not found"
            console.print(f"[WARNING] {feedback}")
            return False, feedback, []
        outputs: List[Dict[str, Any]] = []
        for idx, html in enumerate(html_samples_inner):
            try:
                result = parse_fn(html)
            except Exception as e:
                feedback = (
                    f"error when calling parse_element on sample {idx}: {type(e).__name__}: {e}"
                )
                console.print(f"[WARNING] {feedback}")
                console.print_exception(show_locals=False)
                return False, feedback, []
            ok, msg = _validate_result(result)
            if not ok:
                feedback = f"invalid return value for sample {idx}: {msg}"
                console.print(f"[WARNING] {feedback}")
                return False, feedback, []
            outputs.append(result)
        return True, "success", outputs

    # --- END CHANGE --- #

    console.print("[INFO] starting agent...")

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
           – **CRITICAL**: If a value cannot be extracted or is empty/null, **OMIT the key entirely** from the result dictionary. DO NOT include keys with `None` values.
        2. never raise inside `parse_element`; fail gracefully.
        3. do NOT perform I/O, prints, or network calls. safe on import.
        4. import `BeautifulSoup` and `re` exactly once at the top if needed.

        Start by identifying the values to extract based on the provided HTML examples. Below are some general keys you should always look for.
        However, you should ALWAYS supplement these with additional keys to EXHAUSTIVELY capture all useful information from the elements.
        IMPORTANT: omit keys entirely if no corresponding data is found. Aim for Mutually Exclusive, Collectively Exhaustive (MECE) results – avoid storing the same piece of data under multiple keys.

        GENERAL KEYS TO CONSIDER (adapt and add based on specific html):
        - **URLs**:
            - `primary_url`: The most prominent link (often the permalink/canonical link to the item).
            - `urls`: A list of ALL *other* distinct URLs found (EXCLUDE the `primary_url` from this list).
        - **Identification**:
            - `id`: A stable identifier specifically for the **content item/element itself** (e.g., from a `data-id` attribute, or the unique part of its `primary_url`). **CRITICAL**: This should NOT be the author's handle or user ID.
            - `title`: A primary title or heading (look in `h*`, `title` tags, `aria-label`).
        - **Author Information**:
            - `author`: The display name of the author.
            - `author_url`: The URL to the author's profile.
            - `author_handle`: The author's handle (often near the name/url, sometimes prefixed with a symbol).
            - `author_avatar_url`: URL of the author's avatar/profile image.
        - **Content**:
            - `description`: The main text content. **CRITICAL**: Find the *most specific* HTML element containing the primary body text. Actively **exclude** surrounding metadata like author names/handles, timestamps, "Verified" badges, or action button text (Reply, Like, etc.). Prioritize dedicated text containers (e.g., `<p>`, `article > div + div`) before falling back to broader text extraction. Ensure the final value is *only* the clean body text.
            - `images`: A list of dictionaries for each image, containing `src`, `alt`, `title`, `width`, `height`, and relevant `data-*` attributes. 
        - **Timestamps**:
            - `timestamp`: Human-readable time (e.g., "17h", "May 3"). Look for `<time>` tags or text near author info.
            - `datetime`: Machine-readable timestamp (e.g., ISO 8601 format). Look for the `datetime` attribute on `<time>` tags.
        - **Metrics/Stats** (look for numbers associated with icons or action buttons using stable attributes):
            - `reply_count`, `repost_count`, `like_count`, `bookmark_count`, `view_count`: Parse numerical values (handle 'K'/'M' suffixes if present, converting to integers).
        - **Ranking/Position**:
            - `rank`: Ordinal position if applicable (e.g., in search results). Look for `data-rank` or similar attributes.

        ADVANCED TECHNIQUES & ROBUSTNESS:
        - **Metric Parsing**: When extracting counts (likes, views, etc.), handle 'K'/'M' suffixes (convert 1.2K to 1200, 1M to 1000000) and remove commas.
        - **User Info**: Prioritize stable selectors for user information areas. If the primary method fails, implement fallbacks (e.g., finding links near avatars). Construct `author_url` from the handle if possible. Clean noise like 'Verified account' from names.
        - **Text Formatting**: For the `description`, extract the clean text content. Avoid including complex formatting or trying to convert links within it.
        - **Media Filtering**: When extracting `images`/`videos`, filter out irrelevant items like profile avatars (if `author_avatar_url` is separate), emojis, or `data:` URIs. Use video `poster` attributes for thumbnails.
        - **Metadata Strategy**: For `primary_url` and `datetime`, first try to find a single `<a>` tag containing BOTH the canonical path (like a link to the item itself) AND the `<time datetime=...>` tag. If that fails, find the best candidate canonical link for `primary_url` and the best `<time datetime=...>` for `datetime` separately.
        - **Nested Content**: Look for nested structures (e.g., divs/articles) that indicate quoted/embedded content. If found, try to extract key fields like `quoted_author_name`, `quoted_text`, `quoted_url` using similar logic.

        TOOL USAGE:
        - **CRITICAL**: You MUST evaluate ALL generated or modified Python code using the `evaluate_and_sample_code` tool BEFORE concluding your response.
        - **DO NOT** include Python code directly in your text response. ALWAYS use the tool to provide the code.
        - The tool will return a `CodeEvaluationResult` object containing `success`, `feedback`, `sampled_output_with_html`, and `iteration_count`.
        - If `success` is false, read the `feedback`, fix your code, and **call the tool again** with the corrected code.
        - If `success` is true:
            - Examine the `sampled_output_with_html` (a json string of ONE sample `{html_input: ..., extracted_data: ...}`). Compare the `extracted_data` directly against the `html_input`.
            - CAREFULLY read the `feedback` field – even if success is true, it may contain important quality notes (e.g., missing data, redundant fields).
            - **Mandatory Iteration**: You MUST call the tool AGAIN at least once (i.e., perform at least iteration 2) even if the first attempt (`iteration_count: 1`) succeeded. Use this mandatory iteration to refine your code based on the quality feedback and your own analysis of the paired sample.
            - Continue iterating using the tool if necessary until the code is robust, correct, and addresses all feedback.
        - **Refinement**: In each iteration (especially the mandatory second one), focus on:
            - Fixing any specific errors or quality issues mentioned in the `feedback`.
            - Improving extraction based on comparing the `extracted_data` to the `html_input` in the sample.
            - Ensuring the code adheres to all TASK guidelines (exhaustiveness, robustness, mece).

        FINAL RESPONSE FORMAT:
        - After you have successfully validated the code using the tool (including the mandatory second iteration), your FINAL response MUST be ONLY the raw Python code string itself.
        - DO NOT include markdown fences (```python ... ```) or any other explanatory text in the final response. Just the code.
        """
    ).strip()

    # --- BEGIN CHANGE: Instantiate Agent *before* tool function --- #
    agent = Agent(
        model_config.codegen_model,
        # tools=[], # Tools registered via decorator
        system_prompt=system_prompt,
    )
    # --- END CHANGE --- #

    # --- BEGIN CHANGE: Define Tool Function Inside with Decorator --- #
    @agent.tool(retries=3)  # Apply decorator with retries
    async def evaluate_and_sample_code(
        ctx: RunContext[None], code: str, iteration_count: int
    ) -> CodeEvaluationResult:
        """Evaluates the python code, runs it on samples, and returns feedback with ONE sampled input/output pair."""
        console.print(
            f"[INFO] Tool evaluate_and_sample_code called with iteration_count: {iteration_count}"
        )
        # html_samples accessed directly from outer scope
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        elif code.startswith("```"):
            code = code[len("```") :].strip()
        if code.endswith("```"):
            code = code[: -len("```")].strip()

        # Call the inner _exec_candidate
        success, feedback, outputs = _exec_candidate(code, html_samples)

        if success:
            console.print("[INFO] agent code passed validation")
            quality_feedback: List[str] = []
            if len(outputs) > 1:
                all_keys = set().union(*(d.keys() for d in outputs))
                quality_feedback.extend(validate_empty_columns(outputs, all_keys))
                quality_feedback.extend(validate_identical_columns(outputs, all_keys))
                quality_feedback.extend(validate_text_representation(outputs, html_samples))
                quality_feedback.extend(validate_redundant_key_pairs(outputs, all_keys))
                quality_feedback.extend(validate_cross_key_duplicates(outputs, all_keys))
                quality_feedback.extend(validate_internal_repetition(outputs, all_keys))
                quality_feedback.extend(validate_naive_text_match(outputs, html_samples))
            quality_feedback_str = "\n- ".join(quality_feedback) if quality_feedback else "None"
            if quality_feedback:
                feedback = f"Quality issues detected:\n- {quality_feedback_str}"
            else:
                feedback = (
                    "Code executed successfully with no quality issues detected by validation."
                )
            sampled_outputs_with_indices = sample_items(outputs, sample_size=1)
            sampled_output_str: Optional[str] = None
            if sampled_outputs_with_indices:
                original_index, sample_dict = sampled_outputs_with_indices[0]
                paired_sample = {
                    "html_input": html_samples[original_index],
                    "extracted_data": sample_dict,
                }
                sampled_output_str = json.dumps(paired_sample, indent=2, ensure_ascii=False)
            if iteration_count == 1:
                retry_message = (
                    f"MANDATORY REFINEMENT (Iteration 1 succeeded, but iteration 2 is required):\n"
                    f"Analyze the quality feedback and the sample output provided below.\n"
                    f"Refine your code to address any issues (e.g., missing data, redundancy, exhaustiveness) and call the tool again.\n\n"
                    f"Quality Feedback:\n{feedback}\n\n"
                    f"Sample Input/Output Pair:\n{sampled_output_str}"
                )
                console.print(
                    f"[INFO] Raising ModelRetry: Forcing mandatory iteration 2. Feedback: {retry_message[:500]}..."
                )
                raise ModelRetry(retry_message)
        else:
            console.print(
                f"[WARNING] Raising ModelRetry: Code validation failed. Feedback: {feedback}"
            )
            raise ModelRetry(feedback)
        return CodeEvaluationResult(
            success=success,
            feedback=feedback,
            sampled_output_with_html=sampled_output_str,
            iteration_count=iteration_count,
        )

    # --- END CHANGE --- #

    # --- BEGIN CHANGE: Remove deps from agent.run call --- #
    # tool_deps = ToolDependencies(html_samples=html_samples)
    iteration = 1
    response = await agent.run(
        f"generate the initial python code and evaluate it using the tool (iteration {iteration}).",
        # deps=tool_deps # Removed
    )
    # --- END CHANGE --- #

    final_code_obj = response.output
    console.print("[INFO] agent run finished.")

    # --- BEGIN CHANGE: Attempt to extract code if returned as JSON dict ---
    final_code = ""
    if isinstance(final_code_obj, dict) and len(final_code_obj) == 1:
        console.print(
            "[WARNING] Agent returned a dict, attempting to extract code from its single value."
        )
        try:
            # Extract the value from the single key-value pair
            extracted_value = next(iter(final_code_obj.values()))
            final_code = str(extracted_value)  # Ensure it's a string
            console.print("[INFO] Successfully extracted code string from dict value.")
        except Exception as e:
            console.print(
                f"[ERROR] Failed to extract code from dict: {e}. Falling back to original object."
            )
            # Fallback to the original object if extraction fails
            final_code = str(final_code_obj)
    else:
        # Assume it's already a string or convertible to one
        final_code = str(final_code_obj)
    # --- END CHANGE ---

    # Clean the final output string, removing potential markdown fences
    final_code = final_code.strip()
    if final_code.startswith("```python"):
        final_code = final_code[len("```python") :].strip()
    elif final_code.startswith("```"):
        final_code = final_code[len("```") :].strip()
    if final_code.endswith("```"):
        final_code = final_code[: -len("```")].strip()

    # Run the final validated code one last time to get definitive outputs.
    # Call the module-level _exec_candidate
    success, feedback, final_outputs = _exec_candidate(final_code, html_samples)
    if not success:
        console.print(
            f"[ERROR] INTERNAL ERROR: Agent returned code that failed final validation: {feedback}"
        )
        try:
            console.rule("[bold red]Final Code Validation Failed (Unexpected)[/bold red]")
            syntax = Syntax(
                final_code, "python", theme="monokai", line_numbers=True, word_wrap=True
            )
            console.print(syntax)
            console.rule()
        except Exception as print_err:
            console.print(
                f"[ERROR] Failed to print final error details: {print_err}", style="bold red"
            )
        raise RuntimeError(
            f"agent returned non-working code despite internal validation: {feedback}"
        )
    console.print("[INFO] successfully extracted outputs using agent's final code.")

    # --- BEGIN CHANGE: Remove message printing from here --- #
    # (Messages captured and printed in main)
    # --- END CHANGE --- #

    return final_code, final_outputs


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
        # Reinstate capture_run_messages wrapper AND variable capture
        with capture_run_messages() as messages:
            model_config = ModelConfig()
            code, extracted = await generate_and_test_agent(processed_samples, model_config)
    # --- BEGIN CHANGE: Catch UnexpectedModelBehavior specifically ---
    except UnexpectedModelBehavior as e:
        console.print(f"[bold red]Agent Error (Retries Likely Exceeded):[/bold red] {e}")
        # Print messages when this specific error occurs
        console.rule("Captured Agent Messages (on failure)")
        console.print(messages)
        console.rule()
        return  # Exit after handling
    except Exception as e:
        # General exception handler (messages might not be available here)
        console.print(f"[bold red]failed (General Exception):[/bold red] {e}")
        # Optional: Attempt to print messages if they somehow exist, but might error
        # if 'messages' in locals():
        #    console.rule("Captured Agent Messages (General Failure)")
        #    console.print(messages)
        #    console.rule()
        # --- END CHANGE ---
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

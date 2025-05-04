import json
from pathlib import Path
from typing import Optional

from selectron.ai.selector_tools import SelectorTools
from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

logger = get_logger(__name__)


async def save_debug_elements(
    tools_instance: SelectorTools,
    selector: str,
    selector_description: str,
    url: str,
    reasoning: str,
    output_dir: Path | None = None,  # Use None default
    max_matches_to_detail: Optional[int] = None,  # Changed default to None
) -> None:
    """
    Evaluates a CSS selector, saves matched HTML snippets and metadata to a JSON file for debugging.

    The filename is based on the slugified URL.

    Args:
        tools_instance: An instance of SelectorTools initialized with the relevant HTML.
        selector: The CSS selector to evaluate.
        selector_description: The original description used to generate the selector.
        url: The URL the selector was generated for.
        reasoning: The reasoning provided by the agent for the selector.
        output_dir: The directory where the JSON file should be saved. Defaults to cwd/debug_outputs.
        max_matches_to_detail: The maximum number of elements to get details for.
                                If None, details for ALL matches are included.
    """
    # Set default dir if None
    if output_dir is None:
        output_dir = Path.cwd() / ".selectron"

    # Create the output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create debug output directory {output_dir}: {e}", exc_info=True)
        return  # Cannot proceed without output directory

    # Generate filename from slugified URL
    slug = slugify_url(url)
    filename = f"{slug}.json"
    output_path = output_dir / filename

    try:
        # Evaluate the final selector *again* to get HTML details
        final_eval_result = await tools_instance.evaluate_selector(
            selector=selector,
            target_text_to_check=selector_description,
            return_matched_html=True,
            max_matches_to_detail=max_matches_to_detail,
        )

        html_elements = []
        if final_eval_result and not final_eval_result.error:
            if final_eval_result.matched_html_snippets:
                html_elements = final_eval_result.matched_html_snippets
            else:
                logger.debug(
                    f"DEBUG: Final selector '{selector}' found {final_eval_result.element_count} elements, but no HTML snippets were returned."
                )
        elif final_eval_result and final_eval_result.error:
            logger.error(
                f"DEBUG: Failed to evaluate final selector '{selector}' for debug output. Error: {final_eval_result.error}"
            )
        else:
            logger.error(
                f"DEBUG: Failed to evaluate final selector '{selector}' for debug output. No result object returned."
            )

        # Construct the output data
        output_data = {
            "url": url,
            "selector": selector,
            "selector_description": selector_description,
            "reasoning": reasoning,
            "html_elements": html_elements,
        }

        # Write the data to the JSON file
        try:
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.debug(
                f"DEBUG: Wrote {len(html_elements)} HTML snippets and metadata to {output_path}"
            )
        except Exception as json_write_err:
            logger.error(
                f"DEBUG: Failed to write debug data to JSON: {json_write_err}",
                exc_info=True,
            )

    except Exception as debug_eval_err:
        logger.error(
            f"DEBUG: Exception during final selector evaluation for debug output: {debug_eval_err}",
            exc_info=True,
        )

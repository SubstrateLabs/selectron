import json
import reprlib
from typing import Any, Dict, List

from bs4 import BeautifulSoup

from selectron.util.logger import get_logger

logger = get_logger(__name__)


def execute_parser_on_html(
    html_content: str, selector: str, python_code: str
) -> List[Dict[str, Any]]:
    """
    Executes a parser's Python code against elements matching a selector in static HTML content.

    Args:
        html_content: The static HTML string to parse.
        selector: The CSS selector to find elements.
        python_code: The string containing the Python code for the parser,
                     expected to define a 'parse_element' function.

    Returns:
        A list of dictionaries, where each dictionary is the result of
        calling 'parse_element' on the outer HTML of a matched element.
        Returns an empty list if the selector finds no elements, the Python
        code fails to execute or define 'parse_element', or if 'parse_element'
        fails for all elements.
    """
    results: List[Dict[str, Any]] = []

    # Prepare sandbox for executing the parser's Python code
    sandbox: Dict[str, Any] = {"BeautifulSoup": BeautifulSoup, "json": json}
    try:
        exec(python_code, sandbox)
    except Exception as e:
        logger.error(f"Parser Python code execution error during exec: {e}", exc_info=True)
        return []  # Cannot proceed if the code itself fails

    parse_fn = sandbox.get("parse_element")
    if not callable(parse_fn):
        logger.error("Parser Python code does not define a callable 'parse_element' function.")
        return []  # Cannot proceed without the target function

    # Parse the provided HTML content
    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception as e:
        logger.error(f"Failed to parse provided HTML content: {e}", exc_info=True)
        return []  # Cannot proceed if HTML parsing fails

    # Find elements matching the selector
    try:
        elements = soup.select(selector)
    except Exception as e:
        # Catch errors related to invalid selectors etc.
        logger.error(f"Error applying selector '{selector}' to HTML: {e}", exc_info=True)
        return []  # Cannot proceed if selector fails

    if not elements:
        logger.debug(f"Selector '{selector}' matched no elements in the provided HTML.")
        return []  # Return empty list if no elements match

    # Execute parse_fn on each element's outer HTML
    repr_short = reprlib.Repr()
    repr_short.maxstring = 50
    repr_short.maxother = 50

    for i, element in enumerate(elements):
        try:
            # Convert element back to string to pass to parse_fn
            element_html = str(element)
            # Execute the parse function - no async needed here
            result = parse_fn(element_html)
            if isinstance(result, dict):
                results.append(result)
            else:
                logger.warning(
                    f"parse_element for element {i + 1} returned non-dict type: {type(result).__name__}. Skipping."
                )
        except Exception as e:
            # Log error for the specific element but continue with others
            logger.error(
                f"Error running parse_element for element {i + 1}: {e}",
                exc_info=True,
            )

    logger.debug(f"Executed parser on {len(elements)} elements, got {len(results)} results.")
    return results

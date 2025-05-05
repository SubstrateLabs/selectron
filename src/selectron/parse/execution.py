import json
import reprlib
from typing import Any, Dict, List

from bs4 import BeautifulSoup

from selectron.util.logger import get_logger

from .types import ParseOutcome, ParserError, ParseSuccess

logger = get_logger(__name__)


def execute_parser_on_html(html_content: str, selector: str, python_code: str) -> ParseOutcome:
    """
    Executes a parser's Python code against elements matching a selector in static HTML content.

    Args:
        html_content: The static HTML string to parse.
        selector: The CSS selector to find elements.
        python_code: The string containing the Python code for the parser,
                     expected to define a 'parse_element' function.

    Returns:
        A ParseOutcome object:
        - ParseSuccess(data=[...]) if successful.
        - ParserError(...) if an error occurred during setup or execution.
    """
    results: List[Dict[str, Any]] = []

    # Prepare sandbox for executing the parser's Python code
    sandbox: Dict[str, Any] = {"BeautifulSoup": BeautifulSoup, "json": json}
    try:
        exec(python_code, sandbox)
    except Exception as e:
        msg = f"Parser Python code execution error during exec: {e}"
        logger.error(msg, exc_info=True)
        return ParserError(error_type="python_exec_error", message=msg, details=str(e))

    parse_fn = sandbox.get("parse_element")
    if not callable(parse_fn):
        msg = "Parser Python code does not define a callable 'parse_element' function."
        logger.error(msg)
        return ParserError(error_type="parse_fn_missing", message=msg)

    # Parse the provided HTML content
    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception as e:
        msg = f"Failed to parse provided HTML content: {e}"
        logger.error(msg, exc_info=True)
        return ParserError(error_type="html_parse_error", message=msg, details=str(e))

    # Find elements matching the selector
    try:
        elements = soup.select(selector)
    except Exception as e:
        msg = f"Error applying selector '{selector}' to HTML: {e}"
        logger.error(msg, exc_info=True)
        return ParserError(error_type="selector_error", message=msg, details=str(e))

    if not elements:
        msg = f"Selector '{selector}' matched no elements in the provided HTML."
        logger.debug(msg)
        # Returning success with empty list, as finding nothing isn't necessarily an *error*
        # Caller can decide if empty list is problematic.
        # Alternative: return ParserError(error_type="no_elements_found", message=msg)
        return ParseSuccess(data=[])

    # Execute parse_fn on each element's outer HTML
    repr_short = reprlib.Repr()
    repr_short.maxstring = 50
    repr_short.maxother = 50

    individual_errors = []
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
            err_msg = f"Error running parse_element for element {i + 1}: {e}"
            logger.error(
                err_msg,
                exc_info=True,
            )
            individual_errors.append(err_msg)

    logger.debug(f"Executed parser on {len(elements)} elements, got {len(results)} results.")
    # If *all* elements failed, maybe that should be an error? For now, return successful results.
    # Consider adding logic here if needed.

    return ParseSuccess(data=results)

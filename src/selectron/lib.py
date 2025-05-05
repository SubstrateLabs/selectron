from typing import Any, Dict, List

from selectron.parse.execution import execute_parser_on_html
from selectron.parse.parser_registry import ParserRegistry
from selectron.util.logger import get_logger

logger = get_logger(__name__)

# Initialize the registry once, could be a module-level singleton
# or managed differently if lifecycle needs control.
_parser_registry = ParserRegistry()


def parse(url: str, html_content: str) -> List[Dict[str, Any]]:
    """
    Parses structured data from HTML content based on a registered parser for the URL.

    Finds the most specific parser definition matching the URL (or its parent paths),
    then executes that parser's selector and Python code against the provided HTML.

    Args:
        url: The URL associated with the HTML content, used for parser lookup.
        html_content: The static HTML string to parse.

    Returns:
        A list of dictionaries containing the extracted data, as returned by the
        parser's 'parse_element' function for each matched element. Returns an
        empty list if no suitable parser is found, the parser definition is
        invalid, the selector matches no elements, or parsing fails.
    """
    candidates = _parser_registry.load_parser(url)

    if not candidates:
        logger.info(f"No parser candidates found for URL: {url}")
        return []

    # Use the first candidate (most specific match)
    parser_dict, origin, file_path, matched_slug = candidates[0]
    logger.info(
        f"Using parser '{matched_slug}' (origin: {origin}) found at {file_path} for URL: {url}"
    )

    selector = parser_dict.get("selector")
    python_code = parser_dict.get("python")

    if not selector or not isinstance(selector, str):
        logger.error(f"Parser '{matched_slug}' is missing a valid 'selector'. Cannot parse.")
        return []

    if not python_code or not isinstance(python_code, str):
        logger.error(f"Parser '{matched_slug}' is missing 'python' code. Cannot parse.")
        return []

    # Read the python code if necessary - currently `parser_dict` should have it directly
    # if origin == "source": # Or handle Path vs Traversable if needed, but load_parser gives dict
    #     pass
    # elif origin == "user":
    #     pass
    # For now, assuming python_code in parser_dict is correct.

    try:
        # Execute the parsing logic using the utility function
        results = execute_parser_on_html(
            html_content=html_content,
            selector=selector,
            python_code=python_code,
        )
        return results
    except Exception as e:
        # Catch any unexpected errors during execution call
        logger.error(
            f"Unexpected error executing parser '{matched_slug}' for URL {url}: {e}",
            exc_info=True,
        )
        return []


# Optional: Add a function to allow rescanning parsers if needed by the library user
def rescan_parsers():
    """Forces a rescan of source and user parser directories."""
    logger.info("Forcing parser registry rescan via library call.")
    _parser_registry.rescan_parsers()

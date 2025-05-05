from selectron.parse.execution import execute_parser_on_html
from selectron.parse.parser_registry import ParserRegistry
from selectron.parse.types import ParseOutcome, ParserError
from selectron.util.logger import get_logger

logger = get_logger(__name__)

_parser_registry = ParserRegistry()


def parse(url: str, html_content: str) -> ParseOutcome:
    """
    Parses structured data from HTML content based on a registered parser for the URL.

    Finds the most specific parser definition matching the URL (or its parent paths),
    then executes that parser's selector and Python code against the provided HTML.

    Args:
        url: The URL associated with the HTML content, used for parser lookup.
        html_content: The static HTML string to parse.

    Returns:
        A ParseOutcome object:
        - ParseSuccess(data=[...]) if successful.
        - ParserError(...) if no parser found, parser invalid, or parsing failed.
    """
    try:
        candidates = _parser_registry.load_parser(url)
    except Exception as e:
        msg = f"Error loading parser candidates for URL '{url}': {e}"
        logger.error(msg, exc_info=True)
        return ParserError(error_type="parser_load_error", message=msg, details=str(e))

    if not candidates:
        msg = f"No parser candidates found for URL: {url}"
        logger.info(msg)
        return ParserError(error_type="parser_load_error", message=msg)

    parser_dict, origin, file_path, matched_slug = candidates[0]
    logger.debug(
        f"Using parser '{matched_slug}' (origin: {origin}) found at {file_path} for URL: {url}"
    )

    selector = parser_dict.get("selector")
    python_code = parser_dict.get("python")

    if not selector or not isinstance(selector, str):
        msg = f"Parser '{matched_slug}' is missing a valid 'selector'. Cannot parse."
        logger.error(msg)
        return ParserError(error_type="invalid_parser_definition", message=msg)

    if not python_code or not isinstance(python_code, str):
        msg = f"Parser '{matched_slug}' is missing 'python' code. Cannot parse."
        logger.error(msg)
        return ParserError(error_type="invalid_parser_definition", message=msg)

    try:
        # execution function now returns ParseOutcome directly
        outcome = execute_parser_on_html(
            html_content=html_content,
            selector=selector,
            python_code=python_code,
        )
        return outcome
    except Exception as e:
        # Catch any unexpected errors during execution call
        msg = f"Unexpected error executing parser '{matched_slug}' for URL {url}: {e}"
        logger.error(msg, exc_info=True)
        return ParserError(error_type="unexpected_error", message=msg, details=str(e))

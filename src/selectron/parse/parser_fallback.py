import json
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

# Assuming ParserInfo is defined elsewhere (e.g., in parser_registry.py) correctly
from .parser_registry import ParserInfo, ParserOrigin

logger = get_logger(__name__)


def _load_parser_from_ref(
    ref: Union[Traversable, Path], slug: str, url_for_logging: str
) -> Optional[Dict[str, Any]]:
    """Helper to load and parse JSON content from a Traversable or Path resource."""
    try:
        # Check if it's a Path object (from app dir) or Traversable (from package)
        if isinstance(ref, Path):
            content = ref.read_text(encoding="utf-8")
        elif isinstance(ref, Traversable):
            content = ref.read_text(encoding="utf-8")
        else:
            # This case should ideally not happen if ParserRegistry is correct
            logger.error(f"Invalid parser reference type for slug '{slug}': {type(ref)}")
            return None

        parser_data = json.loads(content)
        # TODO: Add validation for the loaded parser_data structure?
        return parser_data
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from parser file for slug '{slug}' (url '{url_for_logging}')",
            exc_info=True,
        )
        return None
    except OSError as e:
        logger.error(
            f"Error reading parser file for slug '{slug}' (url '{url_for_logging}'): {e}",
            exc_info=True,
        )
        return None
    except Exception as e:  # Catch unexpected errors during loading/parsing
        logger.error(
            f"Unexpected error loading parser for slug '{slug}' (url '{url_for_logging}'): {e}",
            exc_info=True,
        )
        return None


def find_fallback_parser(
    url: str,
    available_parsers: Dict[str, ParserInfo],
) -> Optional[Tuple[Dict[str, Any], ParserOrigin, Path]]:
    """
    Finds a parser for a URL using fallback logic: exact match, parent paths, domain root.

    Args:
        url: The target URL.
        available_parsers: Dictionary mapping slugs to ParserInfo tuples (origin, resource, file_path).

    Returns:
        A tuple containing the parser dict, its origin ('source' or 'user'), and its file path
        if found, otherwise None.
    """
    if not url:
        return None

    parsed_url = urlparse(url)
    url_slug = slugify_url(url)
    domain_slug = slugify_url(f"{parsed_url.scheme}://{parsed_url.netloc}")

    # 1. Check for exact match
    if url_slug in available_parsers:
        origin, resource, file_path = available_parsers[url_slug]
        parser_dict = _load_parser_content(resource)
        if parser_dict:
            logger.debug(
                f"Found exact parser match for '{url}' (slug: '{url_slug}', origin: {origin})"
            )
            return parser_dict, origin, file_path
        else:
            logger.warning(
                f"Found exact slug '{url_slug}' but failed to load content from {resource}"
            )

    # 2. Check for parent path matches
    path_parts = [part for part in parsed_url.path.split("/") if part]
    current_path = ""
    # Iterate from longest parent path to shortest (excluding root)
    for i in range(len(path_parts) - 1, 0, -1):
        current_path = "/" + "/".join(path_parts[:i])
        parent_url = f"{parsed_url.scheme}://{parsed_url.netloc}{current_path}"
        parent_slug = slugify_url(parent_url)
        if parent_slug in available_parsers:
            origin, resource, file_path = available_parsers[parent_slug]
            parser_dict = _load_parser_content(resource)
            if parser_dict:
                logger.debug(
                    f"Found fallback parser match for '{url}' via parent '{parent_url}' (slug: '{parent_slug}', origin: {origin})"
                )
                return parser_dict, origin, file_path
            else:
                logger.warning(
                    f"Found parent slug '{parent_slug}' but failed to load content from {resource}"
                )

    # 3. Check for domain root match (if not already the exact match)
    if url_slug != domain_slug and domain_slug in available_parsers:
        origin, resource, file_path = available_parsers[domain_slug]
        parser_dict = _load_parser_content(resource)
        if parser_dict:
            logger.debug(
                f"Found fallback parser match for '{url}' via domain root '{parsed_url.scheme}://{parsed_url.netloc}' (slug: '{domain_slug}', origin: {origin})"
            )
            return parser_dict, origin, file_path
        else:
            logger.warning(
                f"Found domain slug '{domain_slug}' but failed to load content from {resource}"
            )

    logger.debug(f"No parser found for URL '{url}' after checking fallbacks.")
    return None


def _load_parser_content(resource: Union[Traversable, Path]) -> Optional[Dict[str, Any]]:
    """Loads and parses JSON content from a Traversable or Path resource."""
    try:
        content = resource.read_text(encoding="utf-8")
        return json.loads(content)
    except FileNotFoundError:
        logger.error(f"Parser resource not found: {resource}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from parser resource {resource}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading parser resource {resource}: {e}", exc_info=True)
        return None

import json
from importlib.abc import Traversable
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

logger = get_logger(__name__)


def _load_parser_from_ref(
    ref: Traversable, slug: str, url_for_logging: str
) -> Optional[Dict[str, Any]]:
    """Helper to load and parse JSON content from a Traversable resource."""
    try:
        content = ref.read_text(encoding="utf-8")
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
    url: str, available_parsers: Dict[str, Traversable]
) -> Optional[Dict[str, Any]]:
    """
    Attempts to load a parser for the given URL from the available_parsers dict,
    falling back to parent paths/domain.

    It slugs the full URL and checks the dict first. If not found, it removes the last
    path segment, slugs the new URL, checks again, repeating until only the scheme and netloc
    (domain/authority) remain or a parser is found.

    Args:
        url: The target URL.
        available_parsers: A dictionary mapping URL slugs to Traversable resources.

    Returns:
        The parser definition dictionary if found, otherwise None.
    """
    current_url = url
    parsed_origin = urlparse(current_url)

    # Allow fallback if we have a scheme AND (a netloc OR the scheme is 'file')
    can_fallback = parsed_origin.scheme and (parsed_origin.netloc or parsed_origin.scheme == "file")

    if not can_fallback:
        logger.debug(
            f"URL '{url}' lacks necessary components for fallback (scheme and netloc/file). Trying direct load only."
        )
        # Try direct load just in case
        slug = slugify_url(url)
        parser_ref = available_parsers.get(slug)
        if parser_ref:
            return _load_parser_from_ref(parser_ref, slug, url)
        else:
            return None

    while True:
        current_slug = slugify_url(current_url)
        # logger.debug(f"Fallback attempt: trying slug '{current_slug}' for url '{current_url}'")
        parser_ref = available_parsers.get(current_slug)
        if parser_ref:
            parser_data = _load_parser_from_ref(parser_ref, current_slug, url)
            if parser_data:
                if current_url != url:
                    logger.info(
                        f"Found fallback parser for original url '{url}' using definition for '{current_url}' (slug '{current_slug}')"
                    )
                return parser_data
            # else: Loading failed, _load_parser_from_ref logged error, continue fallback

        # Prepare for next iteration: remove last path segment
        parsed_url = urlparse(current_url)
        path = parsed_url.path

        # Check if we are already at the root or have no path
        is_at_root_slash = path == "/"
        is_empty_path = not path

        if is_at_root_slash or is_empty_path:
            # We just tried the root with a slash or an empty path.
            # If it was root with slash, try root *without* slash before giving up.
            if is_at_root_slash and not is_empty_path:
                root_no_slash_url = urlunparse(
                    (parsed_url.scheme, parsed_url.netloc, "", "", "", "")
                )
                if root_no_slash_url != current_url:
                    root_no_slash_slug = slugify_url(root_no_slash_url)
                    # logger.debug(
                    #     f"Fallback attempt: trying root without slash '{root_no_slash_url}' (slug '{root_no_slash_slug}')"
                    # )
                    parser_ref = available_parsers.get(root_no_slash_slug)
                    if parser_ref:
                        parser_data = _load_parser_from_ref(parser_ref, root_no_slash_slug, url)
                        if parser_data:
                            if root_no_slash_url != url:
                                logger.info(
                                    f"Found fallback parser for original url '{url}' using definition for '{root_no_slash_url}' (slug '{root_no_slash_slug}')"
                                )
                            return parser_data
            # Stop fallback if we've tried root with/without slash or had empty path
            break

        # Remove the last segment
        stripped_path = path.rstrip("/")
        parent_path_segments = stripped_path.split("/")[:-1]

        # Reconstruct parent path
        if not parent_path_segments or (
            len(parent_path_segments) == 1 and parent_path_segments[0] == ""
        ):
            parent_path = "/"
        else:
            parent_path = "/".join(parent_path_segments)
            if path.startswith("/") and not parent_path.startswith("/"):
                parent_path = "/" + parent_path

        # Reconstruct the URL with the parent path
        next_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parent_path, "", "", ""))

        if next_url == current_url:
            logger.warning(f"Fallback URL did not shorten from '{current_url}'. Stopping fallback.")
            break

        current_url = next_url

    # logger.debug(f"No parser found for '{url}' even after fallback attempts.")
    return None

import json
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse, urlunparse

from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

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
    url: str, available_parsers: Dict[str, Union[Traversable, Path]]
) -> Optional[Dict[str, Any]]:
    """
    Attempts to load a parser for the given URL from the available_parsers dict,
    falling back to parent paths/domain.

    It slugs the full URL and checks the dict first. If not found, it removes the last
    path segment, slugs the new URL, checks again, repeating until only the scheme and netloc
    (domain/authority) remain or a parser is found.

    Args:
        url: The target URL.
        available_parsers: A dictionary mapping URL slugs to Traversable or Path resources.

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
    # === Domain-wide Fallback ===
    if can_fallback:
        # logger.debug(
        #     f"Path-based fallback failed for '{url}'. Attempting domain-wide fallback."
        # )
        target_scheme = parsed_origin.scheme
        target_netloc = parsed_origin.netloc

        # Find candidate slugs matching the scheme and netloc
        domain_candidates = []

        # Determine the base slug for the target domain/scheme
        if target_scheme == "file":
            # For file URIs, the effective "domain" starts with file:///
            target_domain_base = slugify_url("file:///")  # Should produce 'file~~3a~~2f~~2f'
        else:
            # For http/https etc., slugify the scheme + netloc
            target_domain_base = slugify_url(
                urlunparse((target_scheme, target_netloc, "", "", "", ""))
            )
            # Handle cases where slugify might simplify http://domain.com to domain.com
            # but we need to match slugs like domain.com~~2fpath
            # Check if the target URL only had scheme and netloc
            is_target_base_only = not parsed_origin.path or parsed_origin.path == "/"
            if is_target_base_only and not target_domain_base.endswith("~~2f") and target_netloc:
                # If the available slugs *might* include a root path slug (domain.com~~2f)
                # we should potentially consider both forms. This is getting complex.
                # Let's stick to the simpler startswith logic for now.
                pass  # Keep base as is for now

        for slug, parser_ref in available_parsers.items():
            # Check if the slug belongs to the same domain/scheme base
            if slug.startswith(target_domain_base):
                # Calculate "path depth" - number of path segments after domain base
                path_part = slug[len(target_domain_base) :]
                # Depth is the count of slugified slashes ('~~2f')
                # Add 1 if path_part is not empty but doesn't start with ~~2f (e.g. domain~~3a8080~~2fpath)
                depth = path_part.count("~~2f")
                if path_part and not path_part.startswith("~~2f") and "~~2f" in path_part:
                    # This handles cases like example.com~~3a8080~~2fpath where the first segment isn't delimited by ~~2f
                    # However, simply counting ~~2f should still give a relative depth measure.
                    # Let's refine the depth logic: count separators + 1 if non-empty path_part exists.
                    pass  # Sticking to simple count for sorting relative depth.

                # Ensure we don't match the exact slug we already failed to find in path fallback
                # (This check might be redundant if path fallback guarantees trying exact first, but safe to keep)
                # if slug == slugify_url(url):
                #      continue

                domain_candidates.append({"slug": slug, "ref": parser_ref, "depth": depth})
            # else: logger.debug(f"Slug '{slug}' does not start with base '{target_domain_base}'")

        if domain_candidates:
            # Sort by depth (ascending) to find the "most root"
            domain_candidates.sort(key=lambda x: x["depth"])
            best_match = domain_candidates[0]
            best_slug = best_match["slug"]
            best_ref = best_match["ref"]

            logger.info(
                f"Using domain-wide fallback for '{url}'. Selected parser "
                f"with slug '{best_slug}' (depth {best_match['depth']})."
            )
            parser_data = _load_parser_from_ref(best_ref, best_slug, url)
            if parser_data:
                return parser_data
            else:
                logger.warning(
                    f"Domain-wide fallback failed: could not load parser from ref for slug '{best_slug}'."
                )
        # else: logger.debug(f"No suitable domain-wide fallback candidates found for '{url}'.")

    # logger.debug(f"No parser found for '{url}' after all fallback attempts.")
    return None

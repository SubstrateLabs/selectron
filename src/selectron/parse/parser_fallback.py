import json
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from selectron.parse.types import ParserInfo, ParserOrigin
from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

logger = get_logger(__name__)


def find_fallback_parser(
    url: str,
    available_parsers: Dict[str, ParserInfo],
) -> List[Tuple[Dict[str, Any], ParserOrigin, Path, str]]:
    """
    Finds all potential fallback parsers for a URL, ordered by preference:
    1. Exact match
    2. Parent path match (longest to shortest)
    3. Sibling path match (at each parent level if parent not found/loaded)
    4. Domain root match (covered implicitly by parent check down to root)

    Only parsers that successfully load are included.

    Args:
        url: The target URL.
        available_parsers: Dictionary mapping slugs to ParserInfo tuples (origin, resource, file_path).

    Returns:
        An ordered list of tuples: (parser_dict, origin, file_path, matched_slug)
        for all successfully loaded candidate parsers. Returns empty list if none found.
    """
    candidates: List[Tuple[Dict[str, Any], ParserOrigin, Path, str]] = []
    found_slugs: set[str] = set()

    if not url:
        return candidates

    parsed_url = urlparse(url)
    url_slug = slugify_url(url)

    # 1. Check for exact match
    if url_slug in available_parsers:
        origin, resource, file_path = available_parsers[url_slug]
        parser_dict = _load_parser_content(resource)
        if parser_dict:
            # logger.debug(
            #     f"Found exact parser match for '{url}' (slug: '{url_slug}', origin: {origin}) - Adding as candidate"
            # )
            candidates.append((parser_dict, origin, file_path, url_slug))
            found_slugs.add(url_slug)
        else:
            logger.warning(
                f"Found exact slug '{url_slug}' but failed to load content from {resource}"
            )
            # Do not return yet, fallback might still find something if exact load fails

    # 2. & 3. Check for parent paths and then siblings at each level
    path_parts = [part for part in parsed_url.path.split("/") if part]
    # Sort available slugs once for deterministic sibling search order (important for tests)
    sorted_available_slugs = sorted(available_parsers.keys())

    # Iterate from immediate parent path level down to the root level
    # Example: /a/b/c -> i=2 (/a/b), i=1 (/a), i=0 (/)
    for i in range(len(path_parts), -1, -1):
        # Construct parent URL for current level
        current_path = "/" + "/".join(path_parts[:i])
        # Handle root case where path is just "/" -> slug needs base domain
        if i == 0:
            parent_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        else:
            parent_url = f"{parsed_url.scheme}://{parsed_url.netloc}{current_path}"

        parent_slug = slugify_url(parent_url)

        # 2a. Check if this direct parent parser exists
        parent_found_and_loaded = False
        if parent_slug in available_parsers and parent_slug not in found_slugs:
            origin, resource, file_path = available_parsers[parent_slug]
            parser_dict = _load_parser_content(resource)
            if parser_dict:
                logger.debug(
                    f"Found fallback parser match for '{url}' via parent '{parent_url}' "
                    f"(slug: '{parent_slug}', origin: {origin}) - Adding as candidate"
                )
                candidates.append((parser_dict, origin, file_path, parent_slug))
                found_slugs.add(parent_slug)
                parent_found_and_loaded = True
            else:
                # Parent exists but failed to load
                logger.warning(
                    f"Found parent slug '{parent_slug}' but failed to load content from {resource}"
                )
                parent_found_and_loaded = False
        elif parent_slug in found_slugs:
            parent_found_and_loaded = True

        # 3a. Check for siblings under this parent level ONLY if parent wasn't successfully found AND loaded
        if not parent_found_and_loaded:
            # Define the prefix siblings must share.
            slug_separator = "~~"
            sibling_prefix = parent_slug + slug_separator
            parent_separator_count = parent_slug.count(slug_separator)

            for slug in sorted_available_slugs:
                # Check if it's a potential sibling:
                # - Starts with the parent slug + slug_separator
                # - Is not the target URL itself (already checked in step 1)
                # - Has exactly one more path component (separator count check)
                # - Has not already been added
                if slug.startswith(sibling_prefix) and slug != url_slug and slug not in found_slugs:
                    slug_separator_count = slug.count(slug_separator)
                    expected_separator_count = parent_separator_count + 1

                    # Only proceed if it's an immediate child (sibling)
                    if slug_separator_count == expected_separator_count:
                        origin, resource, file_path = available_parsers[slug]
                        parser_dict = _load_parser_content(resource)
                        if parser_dict:
                            # URL reconstruction might need adjustment if slug format differs
                            # Keeping the existing best-effort reconstruction for now
                            try:
                                parts = slug.split("-", 2)  # This might be wrong now
                                scheme = parts[0]
                                netloc_parts = parts[1].split("-")
                                netloc = netloc_parts[0] + "//" + netloc_parts[1]
                                path_from_rest = parts[2].replace("-", "/")
                                sibling_url_approx = f"{scheme}://{netloc}/{path_from_rest}"
                            except Exception:
                                sibling_url_approx = f"slug: {slug}"

                            logger.debug(
                                f"Found fallback parser match for '{url}' via sibling "
                                f"under parent '{parent_url}' "
                                f"(sibling approx '{sibling_url_approx}', slug: '{slug}', origin: {origin}) - Adding as candidate"
                            )
                            candidates.append((parser_dict, origin, file_path, slug))
                            found_slugs.add(slug)
                        else:
                            logger.warning(
                                f"Found potential sibling slug '{slug}' under parent '{parent_url}' "
                                f"but failed to load content from {resource}"
                            )

    # If loop completes, return all found and loaded candidates
    if not candidates:
        pass
        # logger.debug(f"No fallback parser candidates found or loaded for URL '{url}'")
    return candidates


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

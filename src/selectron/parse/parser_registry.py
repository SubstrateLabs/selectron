import importlib.resources
from importlib.abc import Traversable
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger

from .parser_fallback import find_fallback_parser
from .types import ParserInfo, ParserOrigin

logger = get_logger(__name__)


class ParserRegistry:
    def __init__(self):
        # Store tuples: (origin, resource_handle, file_path)
        self._available_parsers: Dict[str, ParserInfo] = {}
        self._parser_dir_ref: Optional[Traversable] = None
        self._app_parser_dir: Optional[Path] = None

        # 1. Try to locate the base parser directory within package resources
        try:
            self._parser_dir_ref = importlib.resources.files("selectron").joinpath("parsers")
            if not self._parser_dir_ref.is_dir():
                logger.warning(
                    "Base parser directory 'selectron/parsers' not found or not a directory within package resources."
                )
                self._parser_dir_ref = None
            else:
                logger.info(f"Located base parser directory resource: {self._parser_dir_ref}")

        except ModuleNotFoundError:
            logger.warning("Package 'selectron' not found. Cannot load base parsers.")
            self._parser_dir_ref = None
        except Exception as e:
            logger.error(f"Error accessing package resources for base parsers: {e}", exc_info=True)
            self._parser_dir_ref = None

        # 2. Locate and ensure the user-specific parser directory exists
        try:
            self._app_parser_dir = get_app_dir() / "parsers"
            self._app_parser_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured user parser directory exists: {self._app_parser_dir}")
        except Exception as e:
            logger.error(
                f"Failed to create or access user parser directory {self._app_parser_dir}: {e}",
                exc_info=True,
            )
            self._app_parser_dir = None  # Disable user dir if error

        # Perform initial scan
        self.rescan_parsers()  # Call rescan during init

        logger.info(f"Total available parsers loaded: {len(self._available_parsers)}")

    def load_parser(self, url: str) -> List[Tuple[Dict[str, Any], ParserOrigin, Path, str]]:
        """
        Finds all potential parser candidates for the URL, using fallback logic.

        Uses the find_fallback_parser utility to check the exact URL slug,
        parent path slugs, and siblings against available parsers, returning an ordered list.

        Args:
            url: The target URL.

        Returns:
            An ordered list of tuples: (parser_dict, origin, file_path, matched_slug)
            for all successfully loaded candidate parsers. Returns empty list if none found.
        """
        # Pass the combined dictionary
        return find_fallback_parser(url, self._available_parsers)

    def rescan_parsers(self) -> None:
        """Clears the current parser cache and re-scans source and user directories."""
        self._available_parsers.clear()
        loaded_count = 0
        source_parsers_found = 0
        user_parsers_found = 0

        # Scan source directory (if available) - Lower priority
        if self._parser_dir_ref and self._parser_dir_ref.is_dir():
            try:
                for item_resource in self._parser_dir_ref.iterdir():
                    if item_resource.is_file() and item_resource.name.endswith(".json"):
                        slug = item_resource.name[:-5]
                        try:
                            # Resolve Traversable to a real Path using a context manager
                            with as_file(item_resource) as item_path:
                                if (
                                    slug not in self._available_parsers
                                ):  # Add only if not overridden by user parser later
                                    self._available_parsers[slug] = (
                                        "source",
                                        item_resource,
                                        item_path,
                                    )
                                    source_parsers_found += 1
                        except FileNotFoundError:
                            logger.error(
                                f"Could not resolve source parser resource to a file path (might be inside a package?): {item_resource.name}"
                            )
                        except Exception as resolve_err:
                            logger.error(
                                f"Error resolving source parser resource path {item_resource.name}: {resolve_err}",
                                exc_info=True,
                            )

            except Exception as e:
                logger.error(f"Error scanning source parser directory resource: {e}", exc_info=True)

        # Scan user directory (if available) - Higher priority (overwrites source)
        if self._app_parser_dir and self._app_parser_dir.is_dir():
            try:
                for item_path in self._app_parser_dir.glob("*.json"):
                    if item_path.is_file():
                        slug = item_path.stem  # Use stem to get name without extension
                        # User parsers always override source parsers
                        if (
                            slug in self._available_parsers
                            and self._available_parsers[slug][0] == "source"
                        ):
                            logger.debug(f"User parser '{slug}' is overriding source parser.")
                        self._available_parsers[slug] = (
                            "user",
                            item_path,
                            item_path,
                        )  # resource and path are the same for user
                        user_parsers_found += 1
            except Exception as e:
                logger.error(
                    f"Error scanning user parser directory {self._app_parser_dir}: {e}",
                    exc_info=True,
                )

        loaded_count = len(self._available_parsers)
        logger.info(
            f"Parser rescan complete. Found {loaded_count} total definitions ({source_parsers_found} source, {user_parsers_found} user initially scanned)."
        )

    def delete_source_parser(self, slug: str) -> bool:
        """
        Deletes a parser file from the source parser directory.

        Args:
            slug: The identifier slug of the parser to delete.

        Returns:
            True if the parser was found, identified as 'source', and deleted successfully, False otherwise.
        """
        parser_info = self._available_parsers.get(slug)
        if not parser_info:
            logger.warning(f"Attempted to delete non-existent parser with slug: {slug}")
            return False

        origin, _, file_path = parser_info

        if origin != "source":
            logger.warning(
                f"Attempted to delete a non-source parser '{slug}' (origin: {origin}). Deletion refused."
            )
            return False

        if not file_path.exists():
            logger.error(
                f"Source parser '{slug}' record exists but file path not found: {file_path}. Cannot delete."
            )
            # Optionally remove from registry if file is gone?
            # del self._available_parsers[slug]
            return False

        try:
            file_path.unlink()
            logger.info(f"Successfully deleted source parser file: {file_path}")
            # Remove from registry after successful deletion
            del self._available_parsers[slug]
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete source parser file {file_path} for slug '{slug}': {e}",
                exc_info=True,
            )
            return False

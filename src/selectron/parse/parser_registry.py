import importlib.resources
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, Dict, Optional

from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger

from .parser_fallback import find_fallback_parser

logger = get_logger(__name__)


class ParserRegistry:
    def __init__(self):
        self._available_parsers: Dict[str, Traversable] = {}
        self._parser_dir_ref: Optional[Traversable] = None
        self._app_parser_dir: Optional[Path] = None

        # 1. Load base parsers from package resources
        try:
            self._parser_dir_ref = importlib.resources.files("selectron").joinpath("parsers")
            if self._parser_dir_ref.is_dir():
                logger.info("Using source parser directory for loading.")
            else:
                logger.error(
                    "Parser directory 'selectron/parsers' not found within package resources."
                )
                self._parser_dir_ref = None

        except ModuleNotFoundError:
            logger.warning("Package 'selectron' not found. Parser registry will be empty.")
            self._parser_dir_ref = None
        except Exception as e:
            logger.error(
                f"Error accessing package resources or listing base parsers: {e}", exc_info=True
            )
            self._parser_dir_ref = None

        # 2. Load user-specific parsers from app config directory
        app_parsers_dir = get_app_dir() / "parsers"
        user_parser_count = 0
        if app_parsers_dir.is_dir():
            try:
                for item in app_parsers_dir.iterdir():
                    # Use Path objects directly for user parsers
                    if item.is_file() and item.name.endswith(".json"):
                        slug = item.name[:-5]
                        # Only add if not already present from base parsers
                        if slug not in self._available_parsers:
                            # Store the Path object directly
                            self._available_parsers[slug] = item
                            user_parser_count += 1
                        else:
                            logger.debug(
                                f"Skipping user parser '{slug}', as it already exists as a base parser."
                            )
                if user_parser_count > 0:
                    logger.info(
                        f"Found {user_parser_count} additional user parser definitions in {app_parsers_dir}."
                    )
            except Exception as e:
                logger.error(f"Error listing user parsers in {app_parsers_dir}: {e}", exc_info=True)
        else:
            logger.info(f"User parser directory not found or not a directory: {app_parsers_dir}")

        # Also check and prepare the user-specific parser directory
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

        logger.info(f"Total available parsers (base + user): {len(self._available_parsers)}")

    def load_parser(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to load a parser for the URL, using fallback logic.

        Uses the find_fallback_parser utility to check the exact URL slug,
        parent path slugs, and the domain root slug against available parsers
        (both base and user-specific).

        Args:
            url: The target URL.

        Returns:
            The parser definition dictionary if found (either exact or via fallback),
            otherwise None.
        """
        # The available_parsers dict now contains both base and user parsers
        # Pass the combined dictionary and handle Path/Traversable differences in fallback
        return find_fallback_parser(url, self._available_parsers)

    def rescan_parsers(self) -> None:
        """Clears the current parser cache and re-scans source and user directories."""
        logger.info("Rescanning parser directories...")
        self._available_parsers.clear()
        loaded_count = 0

        # Scan source directory (if available)
        if self._parser_dir_ref and self._parser_dir_ref.is_dir():
            try:
                for item in self._parser_dir_ref.iterdir():
                    if item.is_file() and item.name.endswith(".json"):
                        slug = item.name[:-5]
                        # Source dir has lower priority, only add if not already loaded from user dir
                        if slug not in self._available_parsers:
                            self._available_parsers[slug] = item
                            loaded_count += 1
            except Exception as e:
                logger.error(f"Error scanning source parser directory: {e}", exc_info=True)

        # Scan user directory (if available) - This overrides source dir parsers
        if self._app_parser_dir and self._app_parser_dir.is_dir():
            try:
                for item_path in self._app_parser_dir.glob("*.json"):
                    if item_path.is_file():
                        slug = item_path.stem  # Use stem to get name without extension
                        # User dir has higher priority, potentially overwriting source
                        # Note: We need a Traversable here, not a Path. Reconstruct.
                        # This assumes the user dir is findable via importlib resources mechanism
                        # relative to the app's install location or cwd, which might be brittle.
                        # TODO: Revisit how user parsers are loaded. For now, log a warning.
                        logger.warning(
                            f"Loading user parser '{item_path.name}' by direct path - registry might not handle this robustly."
                        )
                        # Attempt to load as path for now, knowing fallback might break
                        self._available_parsers[slug] = item_path  # Store Path, not Traversable
                        loaded_count += 1
            except Exception as e:
                logger.error(
                    f"Error scanning user parser directory {self._app_parser_dir}: {e}",
                    exc_info=True,
                )

        logger.info(f"Parser rescan complete. Found {len(self._available_parsers)} definitions.")

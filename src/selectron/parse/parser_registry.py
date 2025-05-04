import importlib.resources
from importlib.abc import Traversable
from typing import Any, Dict, Optional

from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger

from .parser_fallback import find_fallback_parser

logger = get_logger(__name__)


class ParserRegistry:
    def __init__(self):
        self._available_parsers: Dict[str, Traversable] = {}
        self._parser_dir_ref: Optional[Traversable] = None

        # 1. Load base parsers from package resources
        try:
            self._parser_dir_ref = importlib.resources.files("selectron").joinpath("parsers")
            if not self._parser_dir_ref.is_dir():
                logger.error(
                    "Base parser directory 'selectron/parsers' not found within package resources."
                )
                self._parser_dir_ref = None
            else:
                for item in self._parser_dir_ref.iterdir():
                    if item.is_file() and item.name.endswith(".json"):
                        slug = item.name[:-5]  # Remove .json extension
                        self._available_parsers[slug] = item
                logger.info(f"Found {len(self._available_parsers)} base parser definitions.")

        except ModuleNotFoundError:
            logger.warning("Package 'selectron' not found. Cannot load base parsers.")
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

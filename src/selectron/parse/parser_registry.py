import importlib.resources
from importlib.abc import Traversable
from typing import Any, Dict, Optional

from selectron.util.logger import get_logger

from .parser_fallback import find_fallback_parser

logger = get_logger(__name__)


class ParserRegistry:
    def __init__(self):
        self._available_parsers: Dict[str, Traversable] = {}
        self._parser_dir_ref: Optional[Traversable] = None
        try:
            self._parser_dir_ref = importlib.resources.files("selectron").joinpath("parsers")
            if not self._parser_dir_ref.is_dir():
                logger.error(
                    "Parser directory 'selectron/parsers' not found within package resources."
                )
                self._parser_dir_ref = None
            else:
                # Pre-scan available parsers
                for item in self._parser_dir_ref.iterdir():
                    if item.is_file() and item.name.endswith(".json"):
                        slug = item.name[:-5]  # Remove .json extension
                        self._available_parsers[slug] = item
                logger.info(f"Found {len(self._available_parsers)} parser definitions.")

        except ModuleNotFoundError:
            logger.warning("Package 'selectron' not found. Parser registry will be empty.")
            self._parser_dir_ref = None
        except Exception as e:
            logger.error(
                f"Error accessing package resources or listing parsers: {e}", exc_info=True
            )
            self._parser_dir_ref = None

    def load_parser(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to load a parser for the URL, using fallback logic.

        Uses the find_fallback_parser utility to check the exact URL slug,
        parent path slugs, and the domain root slug against available parsers.

        Args:
            url: The target URL.

        Returns:
            The parser definition dictionary if found (either exact or via fallback),
            otherwise None.
        """
        # Delegate to the standalone fallback function, passing the available parsers map
        return find_fallback_parser(url, self._available_parsers)

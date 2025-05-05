from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

# Where a parser definition originated from
ParserOrigin = Literal["source", "user"]

# Tuple stored internally: (Origin, Handle (Path or Traversable), Resolved Path)
ParserInfo = Tuple[ParserOrigin, Union[Path, Traversable], Path]


ParserErrorType = Literal[
    "python_exec_error",
    "parse_fn_missing",
    "html_parse_error",
    "selector_error",
    "no_elements_found",
    "parser_load_error",
    "invalid_parser_definition",
    "unexpected_error",
]


class ParserError(BaseModel):
    """Details about a parser execution failure."""

    status: Literal["error"] = "error"
    error_type: ParserErrorType
    message: str
    details: Optional[str] = None


class ParseSuccess(BaseModel):
    """Container for successful parsing results."""

    status: Literal["success"] = "success"
    data: List[Dict[str, Any]]


# The overall outcome of a parse attempt
ParseOutcome = Union[ParseSuccess, ParserError]

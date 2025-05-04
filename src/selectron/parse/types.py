from importlib.abc import Traversable
from pathlib import Path
from typing import Literal, Tuple, Union

ParserOrigin = Literal["source", "user"]
ParserInfo = Tuple[ParserOrigin, Union[Traversable, Path], Path]  # origin, resource, file_path

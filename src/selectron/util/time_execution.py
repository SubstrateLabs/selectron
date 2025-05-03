import time
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar

from typing_extensions import ParamSpec

from selectron.util.logger import get_logger

logger = get_logger(__name__)


R = TypeVar("R")
P = ParamSpec("P")


def time_execution_sync(additional_text: str = "") -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{additional_text} executed in {execution_time:.2f}s")
            return result

        return wrapper

    return decorator


def time_execution_async(
    additional_text: str = "",
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug(f"{additional_text} started")
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{additional_text} completed in {execution_time:.2f}s")
            return result

        return wrapper

    return decorator

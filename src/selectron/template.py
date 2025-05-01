from .internal.logger import get_logger

logger = get_logger(__name__)


def template():
    logger.info("Hello from selectron template function!")
    return "Hello from selectron!"

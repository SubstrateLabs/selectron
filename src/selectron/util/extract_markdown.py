from markdownify import markdownify

from selectron.util.logger import get_logger

logger = get_logger(__name__)


def extract_markdown(html: str) -> str:
    try:
        md_content = markdownify(html)
        logger.debug("Successfully converted HTML to Markdown using markdownify.")
        return md_content
    except Exception as e:
        logger.error(f"markdownify failed to convert HTML: {e}", exc_info=True)
        raise

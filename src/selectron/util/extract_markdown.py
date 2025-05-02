from enum import Enum, auto
from io import BytesIO

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from markdownify import markdownify

from selectron.util.logger import get_logger

logger = get_logger(__name__)


class MarkdownStrategy(Enum):
    MARKDOWNIFY = auto()
    DOCLING = auto()


def extract_markdown(html: str, strategy: MarkdownStrategy) -> str:
    """Converts HTML content to Markdown using the specified strategy.

    Args:
        html: The HTML content as a string.
        strategy: The conversion strategy to use (MarkdownStrategy.MARKDOWNIFY or MarkdownStrategy.DOCLING).

    Returns:
        The Markdown representation of the HTML.

    Raises:
        ValueError: If an invalid strategy is provided.
        Exception: Propagates exceptions from the underlying libraries.
    """
    logger.info(f"Attempting markdown conversion using strategy: {strategy.name}")

    if strategy == MarkdownStrategy.MARKDOWNIFY:
        try:
            md_content = markdownify(html)
            logger.debug("Successfully converted HTML to Markdown using markdownify.")
            return md_content
        except Exception as e:
            logger.error(f"markdownify failed to convert HTML: {e}", exc_info=True)
            raise

    elif strategy == MarkdownStrategy.DOCLING:
        try:
            html_bytes = html.encode("utf-8")
            # Using a simple placeholder filename as docling might need one
            filename = "input.html"
            in_doc = InputDocument(
                path_or_stream=BytesIO(html_bytes),
                format=InputFormat.HTML,
                backend=HTMLDocumentBackend,
                filename=filename,
            )
            backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(html_bytes))
            dl_doc = backend.convert()
            md_content = dl_doc.export_to_markdown().strip()
            if md_content:
                logger.debug(
                    f"Successfully extracted markdown (len {len(md_content)}) using docling."
                )
            else:
                logger.warning("docling backend produced empty markdown.")
            return md_content
        except Exception as e:
            logger.error(f"docling backend failed: {e}", exc_info=True)
            raise
    else:
        logger.error(f"Invalid markdown strategy provided: {strategy}")
        raise ValueError(f"Invalid markdown strategy: {strategy}")

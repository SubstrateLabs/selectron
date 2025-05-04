import urllib.parse

from bs4 import BeautifulSoup, Tag

from selectron.util.logger import get_logger

logger = get_logger(__name__)


def resolve_urls(html_string: str, base_url: str) -> str:
    """resolve relative href/src attributes to absolute urls using base_url."""
    soup = BeautifulSoup(html_string, "html.parser")

    # resolve hrefs in <a> tags
    for a in soup.find_all("a", href=True):
        try:
            if isinstance(a, Tag) and "href" in a.attrs:
                href = a.attrs["href"]
                if isinstance(href, str) and href:
                    a.attrs["href"] = urllib.parse.urljoin(base_url, href)
        except Exception:
            href_val = a.attrs.get("href", "[missing]") if isinstance(a, Tag) else "[not a tag]"
            logger.warning(
                f"failed to urljoin href '{href_val}' with base '{base_url}'", exc_info=True
            )

    # resolve src in <img> tags
    for img in soup.find_all("img", src=True):
        try:
            if isinstance(img, Tag) and "src" in img.attrs:
                src = img.attrs["src"]
                if isinstance(src, str) and src:
                    img.attrs["src"] = urllib.parse.urljoin(base_url, src)
        except Exception:
            src_val = img.attrs.get("src", "[missing]") if isinstance(img, Tag) else "[not a tag]"
            logger.warning(
                f"failed to urljoin src '{src_val}' with base '{base_url}'",
                exc_info=False,  # reduce noise
            )

    return str(soup)

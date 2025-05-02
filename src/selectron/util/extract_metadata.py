import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, HttpUrl, field_validator

from selectron.util.logger import get_logger

logger = get_logger(__name__)


class HtmlMetadata(BaseModel):
    """Structured metadata extracted from an HTML document."""

    url: Optional[HttpUrl | str] = None  # The final URL after potential redirects (if provided)
    title: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None  # Author of the content
    og_image: Optional[HttpUrl | str] = None
    favicon: Optional[HttpUrl | str] = None
    keywords: Optional[List[str]] = None  # Keywords or tags associated with the content
    published_at: Optional[datetime] = None  # Publication date of the content

    @field_validator("keywords", mode="before")
    def parse_keywords(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            # Convert comma-separated string to list
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


def extract_metadata(html_content: str, url: Optional[str] = None) -> HtmlMetadata:
    """
    Extracts core metadata from HTML content: title, description, og_image and favicon,
    as well as additional identifiers like DOI and arXiv ID.
    """
    soup = BeautifulSoup(html_content, "lxml")  # Use lxml for speed and robustness
    metadata: Dict[str, Any] = {}

    # Add the original URL if provided
    if url:
        metadata["url"] = url

    # --- Title ---
    # Try OpenGraph title first, then regular title tag
    og_title_tag = soup.find("meta", property="og:title")
    title_tag = soup.find("title")

    # First check OG title
    if og_title_tag and isinstance(og_title_tag, Tag):
        og_title_content = og_title_tag.get("content")
        if og_title_content and isinstance(og_title_content, str):
            metadata["title"] = og_title_content.strip()

    # Fall back to regular title if no OG title
    if "title" not in metadata and title_tag and isinstance(title_tag, Tag) and title_tag.string:
        title_text = str(title_tag.string).strip()
        if title_text:
            metadata["title"] = title_text

    # --- Description ---
    # Try OpenGraph description first, then meta description
    og_desc_tag = soup.find("meta", property="og:description")
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})

    # First check OG description
    if og_desc_tag and isinstance(og_desc_tag, Tag):
        og_desc_content = og_desc_tag.get("content")
        if og_desc_content and isinstance(og_desc_content, str):
            metadata["description"] = og_desc_content.strip()

    # Fall back to meta description if no OG description
    if "description" not in metadata and meta_desc_tag and isinstance(meta_desc_tag, Tag):
        meta_desc_content = meta_desc_tag.get("content")
        if meta_desc_content and isinstance(meta_desc_content, str):
            metadata["description"] = meta_desc_content.strip()

    # --- OpenGraph Image ---
    og_image_tag = soup.find("meta", property="og:image")
    if og_image_tag and isinstance(og_image_tag, Tag):
        og_image_content = og_image_tag.get("content")
        if og_image_content and isinstance(og_image_content, str):
            metadata["og_image"] = og_image_content.strip()

    # --- Author ---
    # Check multiple possible locations for author info
    # Order of preference: og:author, article:author, meta author, schema.org,
    # Dublin Core, rel="author", twitter:creator, byline classes
    author_found = False

    # Check OpenGraph author
    og_author_tag = soup.find("meta", property="og:author")
    if og_author_tag and isinstance(og_author_tag, Tag):
        og_author = og_author_tag.get("content")
        if og_author and isinstance(og_author, str) and og_author.strip():
            metadata["author"] = og_author.strip()
            author_found = True

    # Check article:author (common in article schema)
    if not author_found:
        article_author_tag = soup.find("meta", property="article:author")
        if article_author_tag and isinstance(article_author_tag, Tag):
            article_author = article_author_tag.get("content")
            if article_author and isinstance(article_author, str) and article_author.strip():
                metadata["author"] = article_author.strip()
                author_found = True

    # Check meta author tag
    if not author_found:
        meta_author_tag = soup.find("meta", attrs={"name": "author"})
        if meta_author_tag and isinstance(meta_author_tag, Tag):
            meta_author = meta_author_tag.get("content")
            if meta_author and isinstance(meta_author, str) and meta_author.strip():
                metadata["author"] = meta_author.strip()
                author_found = True

    # Check citation_author meta tags (Common in academic sites like arXiv)
    if not author_found:
        citation_author_tags = soup.find_all("meta", attrs={"name": "citation_author"})
        if citation_author_tags:
            authors = []
            for tag in citation_author_tags:
                if isinstance(tag, Tag):
                    author_name = tag.get("content")
                    if author_name and isinstance(author_name, str) and author_name.strip():
                        authors.append(author_name.strip())
            if authors:
                metadata["author"] = ", ".join(authors)  # Join multiple authors
                author_found = True

    # Check schema.org author
    if not author_found:
        # Try <meta itemprop="author">
        schema_meta_author = soup.find("meta", attrs={"itemprop": "author"})
        if schema_meta_author and isinstance(schema_meta_author, Tag):
            schema_author = schema_meta_author.get("content")
            if schema_author and isinstance(schema_author, str) and schema_author.strip():
                metadata["author"] = schema_author.strip()
                author_found = True

        # Try <span itemprop="author">
        if not author_found:
            schema_span_author = soup.find(attrs={"itemprop": "author"})
            if schema_span_author and isinstance(schema_span_author, Tag):
                # It might contain nested elements or direct text
                author_name_elem = schema_span_author.find(attrs={"itemprop": "name"})
                if (
                    author_name_elem
                    and isinstance(author_name_elem, Tag)
                    and author_name_elem.get_text(strip=True)
                ):
                    metadata["author"] = author_name_elem.get_text(strip=True)
                    author_found = True
                elif schema_span_author.get_text(strip=True):
                    metadata["author"] = schema_span_author.get_text(strip=True)
                    author_found = True

    # Check Dublin Core
    if not author_found:
        dc_author_tag = soup.find("meta", attrs={"name": "DC.creator"})
        if dc_author_tag and isinstance(dc_author_tag, Tag):
            dc_author = dc_author_tag.get("content")
            if dc_author and isinstance(dc_author, str) and dc_author.strip():
                metadata["author"] = dc_author.strip()
                author_found = True

    # Check the div.authors structure as a lower priority fallback (less structured)
    if not author_found:
        authors_div = soup.find("div", class_="authors")
        if authors_div and isinstance(authors_div, Tag):
            author_links = authors_div.find_all("a")
            authors = []
            for link in author_links:
                if isinstance(link, Tag):
                    author_name = link.get_text(strip=True)
                    if author_name:
                        authors.append(author_name)
            if authors:
                metadata["author"] = ", ".join(authors)
        rel_author_link = soup.find("link", attrs={"rel": "author"})
        if rel_author_link and isinstance(rel_author_link, Tag):
            rel_author_title = rel_author_link.get("title")
            if rel_author_title and isinstance(rel_author_title, str) and rel_author_title.strip():
                metadata["author"] = rel_author_title.strip()
                author_found = True

        # Also try <a rel="author"> which often contains the author name in the text
        if not author_found:
            a_rel_author = soup.find("a", attrs={"rel": "author"})
            if a_rel_author and isinstance(a_rel_author, Tag) and a_rel_author.get_text(strip=True):
                metadata["author"] = a_rel_author.get_text(strip=True)
                author_found = True

    # As last resort, check twitter:creator
    if not author_found:
        twitter_creator_tag = soup.find("meta", attrs={"name": "twitter:creator"})
        if twitter_creator_tag and isinstance(twitter_creator_tag, Tag):
            twitter_creator = twitter_creator_tag.get("content")
            if twitter_creator and isinstance(twitter_creator, str) and twitter_creator.strip():
                # Remove @ if present (common in Twitter handles)
                creator = twitter_creator.strip()
                if creator.startswith("@"):
                    creator = creator[1:]
                metadata["author"] = creator
                author_found = True

    # Check common byline elements
    if not author_found:
        byline_classes = ["byline", "author", "article-byline", "post-author", "entry-author"]
        for class_name in byline_classes:
            byline_elem = soup.find(class_=class_name)
            if byline_elem and isinstance(byline_elem, Tag):
                if byline_elem.get_text(strip=True):
                    # Direct text in the element
                    author_text = byline_elem.get_text(strip=True)
                    # Clean up common prefixes
                    for prefix in ["By ", "by ", "Author: ", "Written by "]:
                        if author_text.startswith(prefix):
                            author_text = author_text[len(prefix) :]
                    metadata["author"] = author_text.strip()
                    author_found = True
                    break

    # Try to find JSON-LD structured data
    if not author_found:
        import json
        from json.decoder import JSONDecodeError

        json_ld_scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for script in json_ld_scripts:
            if script and isinstance(script, Tag):
                script_text = script.get_text(strip=True)
                if script_text:
                    try:
                        ld_data = json.loads(script_text)
                        # Handle both single objects and arrays
                        if isinstance(ld_data, dict):
                            items = [ld_data]
                        elif isinstance(ld_data, list):
                            items = ld_data
                        else:
                            continue

                        # Look for author in JSON-LD data
                        for item in items:
                            # Check for author property
                            if "author" in item:
                                author_info = item["author"]
                                # Author can be a string or an object with a name
                                if isinstance(author_info, str):
                                    metadata["author"] = author_info.strip()
                                    author_found = True
                                    break
                                elif isinstance(author_info, dict) and "name" in author_info:
                                    metadata["author"] = author_info["name"].strip()
                                    author_found = True
                                    break
                    except (JSONDecodeError, AttributeError):
                        # Skip invalid JSON
                        continue

                    if author_found:
                        break

    # --- Keywords ---
    # Check multiple possible locations for keywords/tags
    # Order of preference: meta keywords, article:tag, og:keywords, news_keywords, schema.org keywords, JSON-LD keywords
    keywords_found = False

    # Check standard meta keywords tag
    meta_keywords_tag = soup.find("meta", attrs={"name": "keywords"})
    if meta_keywords_tag and isinstance(meta_keywords_tag, Tag):
        meta_keywords = meta_keywords_tag.get("content")
        if meta_keywords and isinstance(meta_keywords, str) and meta_keywords.strip():
            metadata["keywords"] = meta_keywords.strip()
            keywords_found = True

    # Check article:tag (used by Facebook)
    if not keywords_found:
        article_tags = soup.find_all("meta", property="article:tag")
        if article_tags:
            tags = []
            for tag in article_tags:
                if isinstance(tag, Tag):
                    tag_content = tag.get("content")
                    if tag_content and isinstance(tag_content, str) and tag_content.strip():
                        tags.append(tag_content.strip())
            if tags:
                metadata["keywords"] = tags
                keywords_found = True

    # Check news_keywords (used by Google News)
    if not keywords_found:
        news_keywords_tag = soup.find("meta", attrs={"name": "news_keywords"})
        if news_keywords_tag and isinstance(news_keywords_tag, Tag):
            news_keywords = news_keywords_tag.get("content")
            if news_keywords and isinstance(news_keywords, str) and news_keywords.strip():
                metadata["keywords"] = news_keywords.strip()
                keywords_found = True

    # Check schema.org keywords
    if not keywords_found:
        schema_keywords = soup.find("meta", attrs={"itemprop": "keywords"})
        if schema_keywords and isinstance(schema_keywords, Tag):
            keywords_content = schema_keywords.get("content")
            if keywords_content and isinstance(keywords_content, str) and keywords_content.strip():
                metadata["keywords"] = keywords_content.strip()
                keywords_found = True

    # Check JSON-LD for keywords
    if not keywords_found:
        import json
        from json.decoder import JSONDecodeError

        json_ld_scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for script in json_ld_scripts:
            if script and isinstance(script, Tag):
                script_text = script.get_text(strip=True)
                if script_text:
                    try:
                        ld_data = json.loads(script_text)
                        # Handle both single objects and arrays
                        if isinstance(ld_data, dict):
                            items = [ld_data]
                        elif isinstance(ld_data, list):
                            items = ld_data
                        else:
                            continue

                        # Look for keywords in JSON-LD data
                        for item in items:
                            if "keywords" in item:
                                keywords_info = item["keywords"]
                                if isinstance(keywords_info, str):
                                    metadata["keywords"] = keywords_info.strip()
                                    keywords_found = True
                                    break
                                elif isinstance(keywords_info, list):
                                    # Filter out empty strings and strip whitespace
                                    valid_keywords = [
                                        k.strip()
                                        for k in keywords_info
                                        if isinstance(k, str) and k.strip()
                                    ]
                                    if valid_keywords:
                                        metadata["keywords"] = valid_keywords
                                        keywords_found = True
                                        break
                    except (JSONDecodeError, AttributeError):
                        continue

                    if keywords_found:
                        break

    # Check arXiv subjects as potential keywords
    if not keywords_found:
        subjects_td = soup.find("td", class_="subjects")
        if subjects_td and isinstance(subjects_td, Tag):
            # Extract text, split by ';', strip whitespace
            subjects_text = subjects_td.get_text(separator=";", strip=True)
            if subjects_text:
                subjects_list = [s.strip() for s in subjects_text.split(";") if s.strip()]
                if subjects_list:
                    # If keywords already exist (e.g., from meta), append; otherwise, set.
                    if "keywords" in metadata:
                        # Combine and deduplicate
                        existing_keywords = metadata["keywords"]
                        if isinstance(existing_keywords, list):
                            metadata["keywords"] = list(
                                dict.fromkeys(existing_keywords + subjects_list)
                            )
                        elif isinstance(
                            existing_keywords, str
                        ):  # Handle case where keywords were a string
                            metadata["keywords"] = list(
                                dict.fromkeys([existing_keywords] + subjects_list)
                            )
                    else:
                        metadata["keywords"] = subjects_list
                    keywords_found = True

    # --- Published Date ---
    # Check multiple possible locations for publication date
    # Order of preference: article:published_time, og:published_time, published_time,
    # datePublished schema.org, DC.date.issued, pubdate, lastmod, date
    date_found = False

    # Check article:published_time (Facebook/OpenGraph)
    og_pub_tag = soup.find("meta", property="article:published_time")
    if og_pub_tag and isinstance(og_pub_tag, Tag):
        og_pub_content = og_pub_tag.get("content")
        if og_pub_content and isinstance(og_pub_content, str) and og_pub_content.strip():
            try:
                metadata["published_at"] = datetime.fromisoformat(og_pub_content.strip())
                date_found = True
            except (ValueError, TypeError):
                pass

    # Check og:published_time
    if not date_found:
        og_pub_tag = soup.find("meta", property="og:published_time")
        if og_pub_tag and isinstance(og_pub_tag, Tag):
            og_pub_content = og_pub_tag.get("content")
            if og_pub_content and isinstance(og_pub_content, str) and og_pub_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(og_pub_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    pass

    # Check published_time
    if not date_found:
        pub_time_tag = soup.find("meta", attrs={"name": "published_time"})
        if pub_time_tag and isinstance(pub_time_tag, Tag):
            pub_time_content = pub_time_tag.get("content")
            if pub_time_content and isinstance(pub_time_content, str) and pub_time_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(pub_time_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    pass

    # Check schema.org datePublished
    if not date_found:
        date_pub_tag = soup.find("meta", attrs={"itemprop": "datePublished"})
        if date_pub_tag and isinstance(date_pub_tag, Tag):
            date_pub_content = date_pub_tag.get("content")
            if date_pub_content and isinstance(date_pub_content, str) and date_pub_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(date_pub_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    pass

    # Check Dublin Core date issued
    if not date_found:
        dc_date_tag = soup.find("meta", attrs={"name": "DC.date.issued"})
        if dc_date_tag and isinstance(dc_date_tag, Tag):
            dc_date_content = dc_date_tag.get("content")
            if dc_date_content and isinstance(dc_date_content, str) and dc_date_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(dc_date_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    try:
                        # Try W3CDTF format (subset of ISO8601)
                        from dateutil import parser

                        metadata["published_at"] = parser.parse(dc_date_content.strip())
                        date_found = True
                    except (ImportError, ValueError, TypeError):
                        pass

    # Check citation_date meta (Common in academic sites)
    if not date_found:
        citation_date_tag = soup.find("meta", attrs={"name": "citation_date"})
        if citation_date_tag and isinstance(citation_date_tag, Tag):
            citation_date_content = citation_date_tag.get("content")
            if (
                citation_date_content
                and isinstance(citation_date_content, str)
                and citation_date_content.strip()
            ):
                try:
                    # Try ISO format first (unlikely for YYYY/MM/DD)
                    metadata["published_at"] = datetime.fromisoformat(citation_date_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    try:
                        # Fallback to dateutil parser
                        from dateutil import parser

                        metadata["published_at"] = parser.parse(citation_date_content.strip())
                        date_found = True
                    except (ImportError, ValueError, TypeError):
                        logger.debug(
                            f"Could not parse citation_date '{citation_date_content}' with dateutil."
                        )
                        pass  # Silently fail if dateutil parsing fails

    # Check pubdate meta
    if not date_found:
        pubdate_tag = soup.find("meta", attrs={"name": "pubdate"})
        if pubdate_tag and isinstance(pubdate_tag, Tag):
            pubdate_content = pubdate_tag.get("content")
            if pubdate_content and isinstance(pubdate_content, str) and pubdate_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(pubdate_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    try:
                        from dateutil import parser

                        metadata["published_at"] = parser.parse(pubdate_content.strip())
                        date_found = True
                    except (ImportError, ValueError, TypeError):
                        pass

    # Check lastmod meta
    if not date_found:
        lastmod_tag = soup.find("meta", attrs={"name": "lastmod"})
        if lastmod_tag and isinstance(lastmod_tag, Tag):
            lastmod_content = lastmod_tag.get("content")
            if lastmod_content and isinstance(lastmod_content, str) and lastmod_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(lastmod_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    try:
                        from dateutil import parser

                        metadata["published_at"] = parser.parse(lastmod_content.strip())
                        date_found = True
                    except (ImportError, ValueError, TypeError):
                        pass

    # Check date meta
    if not date_found:
        date_tag = soup.find("meta", attrs={"name": "date"})
        if date_tag and isinstance(date_tag, Tag):
            date_content = date_tag.get("content")
            if date_content and isinstance(date_content, str) and date_content.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(date_content.strip())
                    date_found = True
                except (ValueError, TypeError):
                    try:
                        from dateutil import parser

                        metadata["published_at"] = parser.parse(date_content.strip())
                        date_found = True
                    except (ImportError, ValueError, TypeError):
                        pass

    # Check date from time tag
    if not date_found:
        time_tags = soup.find_all("time")
        for time_tag in time_tags:
            if not isinstance(time_tag, Tag):
                continue

            # Check for datetime attribute
            datetime_attr = time_tag.get("datetime")
            if datetime_attr and isinstance(datetime_attr, str) and datetime_attr.strip():
                try:
                    metadata["published_at"] = datetime.fromisoformat(datetime_attr.strip())
                    date_found = True
                    break
                except (ValueError, TypeError):
                    try:
                        from dateutil import parser

                        metadata["published_at"] = parser.parse(datetime_attr.strip())
                        date_found = True
                        break
                    except (ImportError, ValueError, TypeError):
                        pass

    # Check JSON-LD for publication date
    if not date_found:
        import json
        from json.decoder import JSONDecodeError

        json_ld_scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for script in json_ld_scripts:
            if script and isinstance(script, Tag):
                script_text = script.get_text(strip=True)
                if script_text:
                    try:
                        ld_data = json.loads(script_text)
                        # Handle both single objects and arrays
                        if isinstance(ld_data, dict):
                            items = [ld_data]
                        elif isinstance(ld_data, list):
                            items = ld_data
                        else:
                            continue

                        # Look for date in JSON-LD data
                        for item in items:
                            for date_field in [
                                "datePublished",
                                "dateCreated",
                                "publishedAt",
                                "dateModified",
                            ]:
                                if date_field in item:
                                    date_str = item[date_field]
                                    if isinstance(date_str, str) and date_str.strip():
                                        try:
                                            metadata["published_at"] = datetime.fromisoformat(
                                                date_str.strip()
                                            )
                                            date_found = True
                                            break
                                        except (ValueError, TypeError):
                                            try:
                                                from dateutil import parser

                                                metadata["published_at"] = parser.parse(
                                                    date_str.strip()
                                                )
                                                date_found = True
                                                break
                                            except (ImportError, ValueError, TypeError):
                                                pass
                            if date_found:
                                break
                    except (JSONDecodeError, AttributeError):
                        continue

                    if date_found:
                        break

    # --- Favicon ---
    # Look for icon in various forms
    for rel in ["icon", "shortcut icon", "apple-touch-icon"]:
        icon_link = soup.find("link", rel=rel)
        if icon_link and isinstance(icon_link, Tag):
            icon_href = icon_link.get("href")
            if icon_href and isinstance(icon_href, str):
                # Basic check to handle root-relative vs absolute favicons
                if url and icon_href.startswith("/") and not icon_href.startswith("//"):
                    parsed_url = urllib.parse.urlparse(url)
                    metadata["favicon"] = urllib.parse.urlunparse(
                        (parsed_url.scheme, parsed_url.netloc, icon_href, "", "", "")
                    )
                else:
                    metadata["favicon"] = icon_href.strip()
                break

    # --- Fallback Logic (if URL is provided) ---
    if url:
        parsed_url = urllib.parse.urlparse(url)
        origin = urllib.parse.urlunparse((parsed_url.scheme, parsed_url.netloc, "", "", "", ""))

        # Fallback for Title
        if not metadata.get("title"):
            # Use last path segment or hostname
            path_segments = [seg for seg in parsed_url.path.split("/") if seg]
            if path_segments:
                fallback_title = path_segments[-1].replace("-", " ").replace("_", " ")
                # Simple attempt to remove common extensions
                for ext in [".html", ".htm", ".php", ".asp", ".aspx", ".jsp"]:
                    if fallback_title.lower().endswith(ext):
                        fallback_title = fallback_title[: -len(ext)]
                        break
                metadata["title"] = fallback_title
            else:
                metadata["title"] = parsed_url.netloc
            logger.debug(f"Using URL-derived fallback title: {metadata['title']}")

        # Fallback for Favicon
        if not metadata.get("favicon"):
            # Construct default favicon URL
            favicon_url = urllib.parse.urljoin(origin, "/favicon.ico")
            metadata["favicon"] = favicon_url
            logger.debug(f"Using URL-derived fallback favicon: {metadata['favicon']}")
        elif (
            metadata.get("favicon")
            and isinstance(metadata["favicon"], str)
            and metadata["favicon"].startswith("/")
        ):
            # Ensure relative favicons found earlier are made absolute
            absolute_favicon = urllib.parse.urljoin(origin, metadata["favicon"])
            if absolute_favicon != metadata["favicon"]:
                logger.debug(
                    f"Converting relative favicon {metadata['favicon']} to absolute {absolute_favicon}"
                )
                metadata["favicon"] = absolute_favicon

    # Validate and return using Pydantic model
    validated_metadata = HtmlMetadata(**metadata)
    return validated_metadata

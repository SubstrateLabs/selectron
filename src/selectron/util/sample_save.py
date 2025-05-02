import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import imagehash
from PIL import Image

from selectron.util.extract_metadata import HtmlMetadata
from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url
from selectron.util.stitch_images import stitch_vertical

logger = get_logger(__name__)

# Store tuples of (Image, hash) to allow deduplication
url_screenshot_data: Dict[str, List[Tuple[Image.Image, imagehash.ImageHash]]] = defaultdict(list)


# --- Helper for Hashing ---
def compute_perceptual_hash(img: Image.Image) -> Optional[imagehash.ImageHash]:
    """Computes the average perceptual hash (aHash) of an image."""
    try:
        # aHash is generally fast and good for this type of deduplication
        img_hash = imagehash.average_hash(img)
        return img_hash
    except Exception as e:
        logger.error(f"Failed to compute image hash: {e}", exc_info=True)
        return None


# --- Path Generation Helper ---
def _get_save_paths(url: str) -> Tuple[Path, str]:
    """Determines the save directory and filename slug based on the URL.

    Args:
        url: The URL of the page.

    Returns:
        A tuple containing (target_directory, path_slug).
    """
    parsed_url = urlparse(url)
    host = parsed_url.netloc.split(":")[0] if parsed_url.netloc else "unknown_host"
    # --- Path Slug Logic ---
    path = parsed_url.path.strip("/")
    path_slug = slugify_url(path) if path else "_root_"  # Slugify only the path
    target_dir = Path("samples") / host / path_slug  # Subdirectory is the path slug
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure the final directory exists
    return target_dir, path_slug


# --- File Saving Helpers ---
def _save_html(html: str, path: Path) -> None:
    """Saves the HTML content to the specified path."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.debug(f"Saved HTML to {path}")
    except OSError as e:
        logger.error(f"OS error saving HTML to {path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving HTML to {path}: {e}", exc_info=True)


def _save_metadata_json(metadata: Optional[HtmlMetadata], path: Path, url_for_logging: str) -> None:
    """Saves the metadata as a JSON file."""
    try:
        if metadata:
            if hasattr(metadata, "model_dump"):
                metadata_dict = metadata.model_dump(mode="json")
            elif hasattr(metadata, "dict"):
                metadata_dict = metadata.dict()
            else:
                logger.error(
                    f"Metadata object for {url_for_logging} lacks .model_dump() or .dict() method. Cannot serialize."
                )
                metadata_dict = {}
        else:
            logger.warning(f"Metadata for {url_for_logging} is None. Saving empty dict.")
            metadata_dict = {}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata to {path}")
    except Exception as json_e:
        logger.error(
            f"Failed to serialize or save metadata for {url_for_logging} to {path}: {json_e}",
            exc_info=True,
        )


def _save_markdown(html: str, path: Path, url_for_logging: str) -> None:
    """Extracts markdown from HTML and saves it to the specified path."""
    try:
        # Assume extract_markdown is available (needs to be imported if not already)
        from selectron.util.extract_markdown import extract_markdown  # Local import

        # 1. Extract original markdown
        original_md_content = extract_markdown(html)
        # 2. Save the original markdown
        with open(path, "w", encoding="utf-8") as f:
            f.write(original_md_content)
        logger.info(f"Saved original markdown (len {len(original_md_content)}) to {path}")
    except ImportError:
        logger.error(
            f"Could not import extract_markdown. Skipping markdown save for {url_for_logging}."
        )
    except Exception as md_e:
        logger.error(
            f"Failed to extract or save raw markdown for {url_for_logging} to {path}: {md_e}",
            exc_info=True,
        )
        # Don't save markdown if extraction/save failed


def _save_dom_string(dom_string: Optional[str], path: Path, url_for_logging: str) -> None:
    """Saves the DOM string to the specified path, if provided."""
    if dom_string is not None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(dom_string)
            logger.info(f"Saved DOM string (len {len(dom_string)}) to {path}")
        except Exception as dom_save_e:
            logger.error(
                f"Failed to save DOM string for {url_for_logging} to {path}: {dom_save_e}",
                exc_info=True,
            )
    else:
        logger.debug(f"No DOM string provided for {url_for_logging}. Skipping save to {path}.")


def _save_stacked_screenshot(image: Optional[Image.Image], path: Path, url: str) -> None:
    """Handles screenshot accumulation, deduplication, stacking, and saving.

    Uses the global `url_screenshot_data` dictionary.
    """
    # Accumulate and Stack Screenshot (with Deduplication)
    if image is None:
        logger.info(
            f"No image provided for {url} in this call. No screenshot saved/stacked to {path}."
        )
        return

    new_hash = compute_perceptual_hash(image)
    if not new_hash:
        logger.warning(f"Could not compute hash for new image ({url}). Skipping append/stack.")
        return

    logger.debug(f"New image for {url} hash: {new_hash}")
    # Get the hash of the last image added for this URL, if any
    last_hash: Optional[imagehash.ImageHash] = None
    if url_screenshot_data[url]:
        last_hash = url_screenshot_data[url][-1][1]

    # Define a threshold for hash difference (hamming distance)
    HASH_DIFF_THRESHOLD = 1

    should_append = True
    if last_hash is not None:
        hash_diff = new_hash - last_hash
        if hash_diff <= HASH_DIFF_THRESHOLD:
            logger.info(
                f"Skipping duplicate image for {url}. Hash diff: {hash_diff} <= {HASH_DIFF_THRESHOLD}"
            )
            should_append = False
        else:
            logger.debug(
                f"Image for {url} is different enough (hash diff: {hash_diff}). Appending."
            )

    if should_append:
        # Append the image and its hash
        url_screenshot_data[url].append((image, new_hash))
        logger.debug(
            f"Appended new unique image for {url}. Total images: {len(url_screenshot_data[url])}"
        )

        # Attempt to stack all *unique* images collected for this URL so far
        current_images = [img for img, _h in url_screenshot_data[url]]  # Extract images from tuples
        logger.info(f"Attempting to stack {len(current_images)} unique images for {url}...")
        try:
            stitched_image = stitch_vertical(current_images)

            if stitched_image:
                stitched_image.save(path, "JPEG", quality=85)  # Save stacked image
                logger.info(f"Saved stacked screenshot (size {stitched_image.size}) to {path}")
            else:
                # stitch_vertical returns None on failure (e.g., incompatible widths)
                logger.warning(
                    f"Stacking failed for {url} (stitch_vertical returned None). Screenshot not saved."
                )
        except Exception as stitch_save_e:
            logger.error(
                f"Error stitching or saving screenshot for {url} to {path}: {stitch_save_e}",
                exc_info=True,
            )


async def save_sample_data(
    *,  # Make args keyword-only
    url: str,
    html: str,
    metadata: HtmlMetadata,
    image: Optional[Image.Image],
    dom_string: Optional[str],
) -> None:
    """Saves HTML, metadata, manages screenshot stacking with deduplication, saves raw markdown, and saves DOM string."""
    try:
        target_dir, path_slug = _get_save_paths(url)

        # Construct full paths
        html_path = target_dir / f"{path_slug}.html"
        json_path = target_dir / f"{path_slug}.json"
        image_path = target_dir / f"{path_slug}.jpg"
        md_path = target_dir / f"{path_slug}.md"
        dom_path = target_dir / f"{path_slug}.dom.txt"  # Path for DOM string

        # --- Save Content --- #
        _save_html(html, html_path)

        # Save Metadata
        _save_metadata_json(metadata, json_path, url)

        # --- Handle Markdown ---
        if html:  # Only process if HTML exists
            _save_markdown(html, md_path, url)
        else:
            logger.warning(f"Skipping markdown save for {url} because HTML is missing.")

        # --- Save DOM String --- #
        _save_dom_string(dom_string, dom_path, url)

        # --- Handle Screenshot --- #
        _save_stacked_screenshot(image, image_path, url)

        # Final confirmation message
        logger.info(f"Saved sample data to {target_dir / path_slug}.[html|json|jpg|md|dom.txt]")

    except OSError as e:
        logger.error(f"OS error saving sample data for {url}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving sample data for {url}: {e}", exc_info=True)

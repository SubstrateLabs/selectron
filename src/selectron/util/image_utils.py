import base64
from pathlib import Path
from typing import List, Optional

from PIL import Image

from selectron.util.logger import get_logger

logger = get_logger(__name__)


def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image file and encodes it as a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        raise
    except Exception:
        logger.error(f"Failed to read or encode image {image_path}", exc_info=True)
        raise


def stitch_vertical(images: List[Image.Image], bg_color=(255, 255, 255)) -> Optional[Image.Image]:
    """Stitches images vertically in the order provided.

    Args:
        images: A list of PIL Image objects in the desired vertical order.
        bg_color: Background color for the canvas.

    Returns:
        A new PIL Image object representing the stitched result, or None if input is empty.
    """
    if not images:
        logger.warning("stitch_vertical called with empty image list.")
        return None

    # No longer need to sort by scrollY
    # images_with_scroll.sort(key=lambda item: item[1])

    # Calculate dimensions
    widths = [img.width for img in images]
    max_width = max(widths) if widths else 0
    total_height = sum(img.height for img in images)  # Simple sum of heights

    if max_width == 0 or total_height == 0:
        logger.warning("Cannot stitch images with zero width or calculated height.")
        return None

    # Create the canvas
    first_image_mode = images[0].mode if images else "RGB"
    if "A" in first_image_mode:
        canvas_mode = "RGBA"
        final_bg_color = bg_color + (255,)
    else:
        canvas_mode = "RGB"
        final_bg_color = bg_color

    stitched_image = Image.new(canvas_mode, (max_width, total_height), final_bg_color)

    # Paste images onto the canvas sequentially
    current_y = 0
    for img in images:
        paste_img = img
        # Ensure image mode matches canvas if necessary
        if paste_img.mode != canvas_mode:
            try:
                if paste_img.mode == "P":
                    paste_img = paste_img.convert("RGBA" if "A" in canvas_mode else "RGB")
                elif "A" in canvas_mode and "A" not in paste_img.mode:
                    paste_img = paste_img.convert("RGBA")
                elif "A" not in canvas_mode and "A" in paste_img.mode:
                    paste_img = paste_img.convert("RGB")
            except Exception as e:
                logger.warning(
                    f"Could not convert image mode {paste_img.mode} to {canvas_mode}: {e}. Skipping paste."
                )
                continue

        paste_x = 0
        paste_y = current_y

        logger.debug(
            f"Pasting image #{images.index(img)} (size {paste_img.size}, mode {paste_img.mode}) at ({paste_x}, {paste_y})"
        )

        mask = None
        if paste_img.mode == "RGBA":
            mask = paste_img.split()[3]

        stitched_image.paste(paste_img, (paste_x, paste_y), mask=mask)
        current_y += img.height  # Move down by the height of the pasted image

    return stitched_image

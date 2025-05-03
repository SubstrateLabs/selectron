import asyncio
import base64
import io
import json
from typing import Optional

import openai
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image

from selectron.util.logger import get_logger

from .types import AutoProposal

DEFAULT_MODEL = "gpt-4.1-nano"

logger = get_logger(__name__)

PROPOSAL_PROMPT = """You are an expert UI analyst. Analyze the provided screenshot.

1.  **Identify the Main Content Area:** Locate the region(s) displaying the core content, ignoring global elements like headers, footers, navigation, and sidebars. Focus on information-rich primary content suitable for extracting important structured data from the page. Categorize the page into one of two types:
    *   (1) Recurring items: A list/feed/grid on the page displaying recurring items (e.g., posts, products, videos, comments).
    *   (2) Single block of content: A single block of content (e.g., an article, a video, a single post)

2.  **Generate Description:** Provide ONE concise, generic description suitable for selecting the **best** main content region based on the page type:
    *   (1) For recurring items, describe the container (e.g., "All posts in the main feed", "All listing items in the search results", "All comments in the comments section").
    *   (2) For single blocks, describe the primary section containing informational text content, focusing on metadata like title, author, date, etc. Examples: "The container with the video title, author, and description", "The article heading containing the title, author, and date".
        *   Important: DO NOT select the main content, your goal is to select the best region for extracting metadata.

Output ONLY a JSON object with a single key "description": `{"description": "Your description here"}`. No other text, labels, formatting, or explanation."""


async def propose_select(
    client: openai.AsyncOpenAI,
    screenshot: Image.Image,
    model: str = DEFAULT_MODEL,
) -> Optional[AutoProposal]:
    """
    Analyzes a screenshot using an OpenAI vision model to propose a UI description
    for potentially recurring elements.

    Args:
        client: An initialized AsyncOpenAI client.
        screenshot: The PIL Image object of the tab content.
        model: The OpenAI model to use (defaults to "gpt-4.1-nano").
        tab_id_for_logging: An optional identifier for logging purposes.

    Returns:
        An AutoProposal object if successful, None otherwise.
    """
    try:
        # 1. Encode image
        buffered = io.BytesIO()
        img_to_save = screenshot
        if img_to_save.mode == "RGBA":
            img_to_save = img_to_save.convert("RGB")
        img_to_save.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 2. Construct messages
        messages: list[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROPOSAL_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]

        # 3. Make API call
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        await asyncio.sleep(0)  # Yield control briefly

        # 4. Parse and validate response
        response_content = completion.choices[0].message.content
        if not response_content:
            logger.warning("Received empty response content from proposal model.")
            return None

        try:
            proposal_data = json.loads(response_content)
            description = proposal_data.get("description")

            if isinstance(description, str) and description.strip():
                return AutoProposal(proposed_description=description.strip())
            else:
                # Handle cases where 'description' is missing, not a string, or empty
                err_msg = "Model response missing valid 'description' field."
                if "description" in proposal_data:
                    err_msg += f" (Type: {type(description)}, Value: '{description}')"
                else:
                    err_msg += " (Field missing)"
                logger.warning(err_msg)
                return None

        except json.JSONDecodeError as json_err:
            logger.error(
                f"Failed to parse JSON response: {json_err}\nResponse: {response_content}",
            )
            return None
    except openai.APIError as api_err:
        logger.error(f"OpenAI API Error during proposal: {api_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during proposal generation: {e}", exc_info=True)
        return None

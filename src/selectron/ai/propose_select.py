import asyncio
import io
from typing import Optional

import openai
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError

from selectron.util.logger import get_logger
from selectron.util.time_execution import time_execution_async

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


class _ProposalResponse(BaseModel):
    """Internal model for parsing the LLM response."""

    description: str = Field(..., description="The proposed description for the main content area")


@time_execution_async("propose_selection")
async def propose_selection(
    screenshot: Image.Image,
) -> Optional[AutoProposal]:
    """
    Analyzes a screenshot using PydanticAI with an OpenAI vision model
    to propose a UI description for potentially recurring elements.

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
        # Ensure image is RGB for JPEG saving
        if img_to_save.mode == "RGBA":
            img_to_save = img_to_save.convert("RGB")
        img_to_save.save(buffered, format="JPEG", quality=85)
        image_bytes = buffered.getvalue()

        # 2. Prepare agent input using BinaryContent
        agent_input = [
            PROPOSAL_PROMPT,
            BinaryContent(data=image_bytes, media_type="image/jpeg"),
        ]

        # 3. Instantiate PydanticAI Agent and make the call
        agent = Agent[None, _ProposalResponse](
            # Pass model name directly
            model=DEFAULT_MODEL,  # TODO: openai/anthropic model
            # Specify the expected structured output type
            output_type=_ProposalResponse,
            # Set max_tokens if needed (might be inferred or have defaults)
            # max_tokens=500, # <-- Uncomment if needed and supported
            # json_mode is implied by specifying output_type
        )

        # Run the agent asynchronously
        result = await agent.run(agent_input)

        await asyncio.sleep(0)  # Yield control briefly

        # 4. Access the validated output
        proposal_response = result.output

        if proposal_response and proposal_response.description:
            return AutoProposal(proposed_description=proposal_response.description.strip())
        else:
            # This case should technically be less likely now due to Pydantic validation,
            # but good to keep as a fallback.
            logger.warning("PydanticAI returned a valid structure but with an empty description.")
            return None

    except openai.APIError as api_err:  # Keep original OpenAI error handling
        logger.error(f"OpenAI API Error during proposal: {api_err}", exc_info=True)
        return None
    except ModelHTTPError as pydantic_ai_err:  # Catch PydanticAI HTTP errors
        logger.error(
            f"PydanticAI ModelError during proposal generation: {pydantic_ai_err}",
            exc_info=True,
        )
        return None
    except Exception as e:  # Catch other potential errors
        logger.error(f"Unexpected error during proposal generation: {e}", exc_info=True)
        return None

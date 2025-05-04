import asyncio
import io
from typing import Optional

from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError

from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig
from selectron.util.time_execution import time_execution_async

from .types import AutoProposal

logger = get_logger(__name__)

PROPOSAL_PROMPT = """You are an expert UI analyst. Analyze the provided screenshot.

1.  **Identify the Main Content Area:** Locate the region(s) displaying the core content, ignoring global elements like headers, footers, navigation, and sidebars. Focus on information-rich primary content suitable for extracting important structured data from the page. Categorize the page into one of two types:
    *   (1) Recurring items: A list/feed/grid on the page displaying recurring items (e.g., posts, products, videos, comments).
    *   (2) Single block of content: A single block of content (e.g., an article, a video, a single post)

2.  **Generate Description:** Provide ONE concise, generic description suitable for selecting the **best** main content region based on the page type:
    *   (1) For recurring items, describe the container (e.g., "All posts in the main feed", "All listing items in the search results", "All comments in the comments section").
    *   (2) For single blocks, describe the primary section containing informational text content, focusing on metadata like title, author, date, etc. 
        *   Important: DO NOT select the main content, your goal is to select the best region for extracting metadata.

Output ONLY a JSON object with a single key "description": `{"description": "Your description here"}`. No other text, labels, formatting, or explanation."""


class _ProposalResponse(BaseModel):
    description: str = Field(..., description="The proposed description for the main content area")


@time_execution_async("propose_selection")
async def propose_selection(
    screenshot: Image.Image,
    model_config: ModelConfig,
) -> Optional[AutoProposal]:
    try:
        buffered = io.BytesIO()
        img_to_save = screenshot
        if img_to_save.mode == "RGBA":
            img_to_save = img_to_save.convert("RGB")
        img_to_save.save(buffered, format="JPEG", quality=85)
        image_bytes = buffered.getvalue()
        agent_input = [
            PROPOSAL_PROMPT,
            BinaryContent(data=image_bytes, media_type="image/jpeg"),
        ]
        agent = Agent[None, _ProposalResponse](
            model=model_config.analyze_model,
            output_type=_ProposalResponse,
        )
        result = await agent.run(agent_input)
        await asyncio.sleep(0)  # Yield control briefly
        proposal_response = result.output
        if proposal_response and proposal_response.description:
            return AutoProposal(proposed_description=proposal_response.description.strip())
        else:
            logger.warning("PydanticAI returned a valid structure but with an empty description.")
            return None

    except ModelHTTPError as pydantic_ai_err:
        logger.error(
            f"PydanticAI ModelError during proposal generation: {pydantic_ai_err}",
            exc_info=True,
        )
        return None
    except Exception as e:  # Catch other potential errors
        logger.error(f"Unexpected error during proposal generation: {e}", exc_info=True)
        return None

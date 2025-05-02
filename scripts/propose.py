import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import openai
from rich.console import Console
from rich.pretty import pretty_repr

from selectron.ai.analyze import generate_extraction_proposal
from selectron.ai.analyze_types import ExtractionProposal
from selectron.util.logger import get_logger

SAMPLE_HOST = "www.linkedin.com"
SAMPLE_SLUG = "feed"

console = Console()
logger = get_logger(__name__)


async def main():
    # Construct path from constants
    image_path = Path(f"samples/{SAMPLE_HOST}/{SAMPLE_SLUG}/{SAMPLE_SLUG}.jpg")
    markdown_path = image_path.with_suffix(".md")

    if not image_path.exists():
        logger.error(f"Image not found at {image_path}, exiting.")
        return

    logger.info(f"Target image: {image_path}")
    logger.info(f"Target markdown: {markdown_path}")

    # Initialize OpenAI client
    try:
        client = openai.AsyncOpenAI()
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        return

    # Call the abstracted function
    proposal: Optional[ExtractionProposal] = await generate_extraction_proposal(
        image_path=image_path,
        markdown_path=markdown_path,
        client=client,
    )

    # Check if the proposal exists AND has items *after* internal filtering
    if proposal and proposal.items:
        console.print("\n[bold magenta]=== Proposed Extractions ===[/bold magenta]")
        # Convert Pydantic model to dict for pretty printing and saving
        proposal_dict = proposal.model_dump(mode="json")
        console.print(pretty_repr(proposal_dict, indent_size=2))

        # --- Save to JSON ---
        try:
            output_dir = image_path.parent
            output_path = output_dir / "proposal.json"
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(proposal_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Proposal saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save proposal to JSON: {e}", exc_info=True)

    elif proposal:  # Proposal exists but items list is empty after filtering
        logger.warning(
            "No items with metadata found after processing. Proposal not printed or saved."
        )
    else:  # generate_extraction_proposal returned None
        logger.error("Failed to generate extraction proposal.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        # Use logger instead of console for this check
        logger.error("OPENAI_API_KEY environment variable not set.")
    else:
        asyncio.run(main())

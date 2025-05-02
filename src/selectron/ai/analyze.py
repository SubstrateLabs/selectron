from pathlib import Path
from typing import Optional

import openai

from selectron.ai.analyze_types import (
    ExtractionProposal,
    MarkdownMappingResponse,
)
from selectron.util.image_utils import encode_image_to_base64
from selectron.util.logger import get_logger

logger = get_logger(__name__)

# Define default models as constants
DEFAULT_IMAGE_ANALYSIS_MODEL = "gpt-4.1"
DEFAULT_MARKDOWN_MAPPING_MODEL = "gpt-4.1"
DEFAULT_MAX_TOKENS = 4096  # Default max tokens


async def generate_extraction_proposal(
    *,  # Enforce keyword arguments
    image_path: Path,
    markdown_path: Path,
    client: openai.AsyncOpenAI,
) -> Optional[ExtractionProposal]:
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception:
        logger.error("Failed to encode image, cannot proceed with analysis.")
        return None
    image_url = f"data:image/jpeg;base64,{base64_image}"
    image_analysis_prompt = (
        "You are an expert analyst specializing in understanding webpage content from screenshots."
        "\nAnalyze the provided image, which is a screenshot of a webpage."
        "\nIdentify the major functional or informational regions visible. **Be granular.**"
        "\n**Important:**"
        "\n1. If you identify a region that contains a list or feed of similar items (like articles, products, posts), treat EACH item in the list as a SEPARATE region."
        "\n2. **AVOID** identifying overly broad, top-level containers like 'main content area', 'body', or 'page wrapper' as single regions. Instead, identify the distinct sections *within* those areas (e.g., 'featured posts section', 'latest articles feed', 'sidebar', 'user profile block'). Focus on semantically meaningful blocks."
        "\nFor each granular region (or individual list item), provide:"
        "\n1. `region_description`: A clear description of the region or list item."
        "\n2. `observed_content_summary`: A brief summary of the actual text or content observed within that region/item in the image."
        "\nDo NOT suggest CSS selectors or HTML element names. Focus ONLY on the visible content and structure."
        "\nRespond using the structure defined by the provided `ExtractionProposal` Pydantic model."
    )
    initial_proposal: Optional[ExtractionProposal] = None
    try:
        response = await client.beta.chat.completions.parse(
            model=DEFAULT_IMAGE_ANALYSIS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            response_format=ExtractionProposal,
            max_tokens=DEFAULT_MAX_TOKENS,  # Use constant
        )
        initial_proposal = response.choices[0].message.parsed
        if not initial_proposal:
            logger.error("empty parsed content.")
            return None
        # Assign sequential IDs
        if initial_proposal.items:
            for i, item in enumerate(initial_proposal.items):
                item.id = i + 1
            logger.info(
                f"Assigned sequential IDs to {len(initial_proposal.items)} proposed regions."
            )
        else:
            logger.warning("Initial proposal contains no items.")

    except openai.APIConnectionError as e:
        logger.error(f"Analysis Service Error: Failed to connect: {e}")
        return None
    except openai.RateLimitError as e:
        logger.error(f"Analysis Service Error: Rate limit exceeded: {e}")
        return None
    except openai.APIStatusError as e:
        logger.error(f"Analysis Service Error: Status {e.status_code}: {e.response}")
        return None
    except Exception:
        logger.error("Unexpected error during image analysis", exc_info=True)
        return None

    if not initial_proposal or not initial_proposal.items:
        logger.warning("empty proposal.")
        return initial_proposal

    markdown_content: Optional[str] = None
    if markdown_path.exists():
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        except Exception:
            logger.warning(
                f"Failed to read markdown file {markdown_path}. Proceeding without markdown content.",
                exc_info=True,
            )
            markdown_content = None
    else:
        logger.warning(f"markdown file not found at {markdown_path}.")
        return initial_proposal

    if not markdown_content:
        logger.warning("markdown content is empty or could not be read.")
        return initial_proposal

    region_descriptions_for_prompt = []
    for i, item in enumerate(initial_proposal.items):
        region_descriptions_for_prompt.append(
            f"{i + 1}. Region ID: {item.id}\n   Description: {item.region_description}\n   Observed Content: {item.observed_content_summary}"
        )
    descriptions_text = "\n\n".join(region_descriptions_for_prompt)

    num_items = len(initial_proposal.items)
    markdown_mapping_prompt = f"""You are a meticulous analyst correlating webpage regions with markdown content and extracting factual metadata.

The full markdown content is below:
--- MARKDOWN START ---
{markdown_content}
--- MARKDOWN END ---

Here are the {num_items} regions previously identified from a screenshot, including descriptions based on visual analysis:
--- IDENTIFIED REGIONS START ---
{descriptions_text}
--- IDENTIFIED REGIONS END ---

Your task is twofold for EACH of the {num_items} identified regions:
1.  **Markdown Snippet:** Locate the most relevant text snippet *within the markdown content* that corresponds strictly to the region's description and observed content.
2.  **Metadata:** Generate a key-value dictionary of specific, factual metadata *strictly based on verifiable information*. Examine the region's description/summary (derived from the image) AND the corresponding markdown snippet. Extract ONLY discrete metadata points explicitly present (e.g., 'author', 'date', 'likes', 'comments', 'read_time', 'status', 'link_url', image alt text). **DO NOT include summaries or descriptions of the main content within the metadata dictionary.** The `markdown_snippet` field is for the content itself.

Respond with a JSON object adhering precisely to the `MarkdownMappingResponse` schema. This object must contain two keys:
*   `mapped_items`: A list containing exactly {num_items} items. Each item bundles the markdown snippet and metadata for that region.

Maintain the exact order from the input regions in both lists. Prioritize accuracy and avoid hallucination; only report what is demonstrably present."""

    try:
        mapping_response = await client.beta.chat.completions.parse(
            model=DEFAULT_MARKDOWN_MAPPING_MODEL,
            messages=[{"role": "user", "content": markdown_mapping_prompt}],
            response_format=MarkdownMappingResponse,
            max_tokens=DEFAULT_MAX_TOKENS,  # Use constant
        )

        markdown_mapping = mapping_response.choices[0].message.parsed
        if (
            not markdown_mapping
            or markdown_mapping.mapped_items is None
            or len(markdown_mapping.mapped_items) != num_items
        ):
            logger.error(
                "Markdown mapping call returned invalid or incomplete content (length mismatch or missing items)."
            )
            # Return original proposal without merging partial/invalid mapping
            return initial_proposal

        merged_count = 0
        for i, mapping_item in enumerate(markdown_mapping.mapped_items):
            proposal_item = initial_proposal.items[i]
            # Use None if the snippet is empty or just whitespace
            proposal_item.markdown_content = (
                mapping_item.markdown_snippet.strip()
                if mapping_item.markdown_snippet and mapping_item.markdown_snippet.strip()
                else None
            )
            # Store metadata if it's a non-empty dictionary
            proposal_item.metadata = mapping_item.metadata if mapping_item.metadata else None
            if proposal_item.markdown_content or proposal_item.metadata:
                merged_count += 1
        logger.info(f"merged markdown/metadata for {merged_count} items.")

    except openai.APIConnectionError as e:
        logger.error(f"Mapping Service Error: Failed to connect: {e}")
    except openai.RateLimitError as e:
        logger.error(f"Mapping Service Error: Rate limit exceeded: {e}")
    except openai.APIStatusError as e:
        logger.error(f"Mapping Service Error: Status {e.status_code}: {e.response}")
    except Exception:
        logger.error("Unexpected error during markdown mapping", exc_info=True)

    # Always filter, even if mapping failed partially or completely
    if initial_proposal and initial_proposal.items:
        original_count = len(initial_proposal.items)
        initial_proposal.items = [
            item for item in initial_proposal.items if item.metadata is not None
        ]  # Keep only if metadata was successfully extracted/merged
        filtered_count = len(initial_proposal.items)
        if original_count != filtered_count:
            logger.info(
                f"filtered out {original_count - filtered_count} items lacking extracted metadata."
            )
    else:
        logger.warning("proposal object or items list is unexpectedly empty before filtering.")
        return None  # Return None if the proposal became invalid somehow

    return initial_proposal

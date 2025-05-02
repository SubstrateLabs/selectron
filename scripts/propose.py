import asyncio
import base64
import json
import os
from pathlib import Path
from typing import List, Optional

import openai
from pydantic import BaseModel, Field
from rich.console import Console
from rich.pretty import pretty_repr

SAMPLE_HOST = "www.linkedin.com"
SAMPLE_SLUG = "in~~2frobertkcheung"


IMAGE_ANALYSIS_MODEL = "gpt-4.1"
MARKDOWN_MAPPING_MODEL = "gpt-4.1"
MAX_TOKENS = 4096


class ProposedContentRegion(BaseModel):
    id: Optional[int] = Field(
        None, description="Unique identifier assigned after initial image analysis."
    )
    region_description: str = Field(
        ...,
        description="Clear description of the major content region identified in the image (e.g., 'Main article content', 'Header navigation', 'Featured posts section').",
    )
    observed_content_summary: str = Field(
        ...,
        description="A brief summary or key text observed within this region in the image. Focus on the actual information visible.",
    )
    markdown_content: Optional[str] = Field(
        None, description="The corresponding markdown content for this region, if found."
    )
    metadata: Optional[dict[str, str]] = Field(
        None,
        description="Key-value pairs of metadata extracted from the markdown content (e.g., 'author', 'date', 'likes').",
    )


class ExtractionProposal(BaseModel):
    items: List[ProposedContentRegion] = Field(
        ..., description="List of proposed content regions identified in the webpage screenshot."
    )


class MarkdownMappingItem(BaseModel):
    markdown_snippet: Optional[str] = Field(
        None,
        description="The corresponding markdown snippet for the region, or null if none found.",
    )
    metadata: Optional[dict[str, str]] = Field(
        None, description="Extracted factual metadata dictionary, or null/empty if none found."
    )


class MarkdownMappingResponse(BaseModel):
    mapped_items: List[MarkdownMappingItem] = Field(
        ...,
        description="A list containing exactly one item for each input region, in the original order. Each item bundles the markdown snippet and metadata for that region.",
    )


# --- Globals ---
console = Console()


# --- Helper Functions ---
def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image file and encodes it as a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Image file not found at {image_path}")
        raise
    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to read or encode image {image_path}: {e}"
        )
        raise


# --- Main Logic ---


async def propose_extractions(image_path: Path) -> Optional[ExtractionProposal]:
    """Analyzes an image using OpenAI vision and proposes structured extractions."""
    console.print(f"Encoding image {image_path}...", style="dim")
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception:
        return None  # Error already printed by encode_image_to_base64

    image_url = f"data:image/jpeg;base64,{base64_image}"

    console.print("Initializing analysis client...", style="dim")
    try:
        # Using instructor requires patching the client
        client = openai.AsyncOpenAI()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to initialize analysis client: {e}")
        return None

    # --- Step 1: Image Analysis ---
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

    console.print("Requesting image analysis...", style="dim")
    initial_proposal: Optional[ExtractionProposal] = None
    try:
        response = await client.beta.chat.completions.parse(
            model=IMAGE_ANALYSIS_MODEL,
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
            max_tokens=MAX_TOKENS,
        )

        initial_proposal = response.choices[0].message.parsed
        if not initial_proposal:
            console.print(
                "[bold red]Error:[/bold red] Analysis service returned empty parsed content."
            )
            # Decide if we should return None or continue without markdown
            return None  # Let's return None if the primary step fails

        console.print("[bold green]Success:[/bold green] Received initial proposal.", style="dim")

        # --- Assign sequential IDs ---
        if initial_proposal and initial_proposal.items:
            for i, item in enumerate(initial_proposal.items):
                item.id = i + 1  # Assign 1-based IDs
            console.print(
                f"Assigned IDs to {len(initial_proposal.items)} proposed regions.", style="dim"
            )

    except openai.APIConnectionError as e:
        console.print(f"[bold red]Analysis Service Error:[/bold red] Failed to connect: {e}")
        return None
    except openai.RateLimitError as e:
        console.print(f"[bold red]Analysis Service Error:[/bold red] Rate limit exceeded: {e}")
        return None
    except openai.APIStatusError as e:
        console.print(
            f"[bold red]Analysis Service Error:[/bold red] Status {e.status_code}: {e.response}"
        )
        return None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Unexpected error during image analysis: {e}")
        return None

    # --- Step 2: Markdown Content Mapping ---
    if not initial_proposal or not initial_proposal.items:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Initial proposal is empty. Skipping markdown mapping."
        )
        return initial_proposal

    markdown_path = image_path.with_suffix(".md")
    markdown_content = None
    if markdown_path.exists():
        console.print(
            f"Attempting to read corresponding markdown file: {markdown_path}", style="dim"
        )
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            console.print("Successfully read markdown file.", style="dim")
        except Exception as e:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Failed to read markdown file {markdown_path}: {e}. Proceeding without markdown content."
            )
            markdown_content = None  # Ensure it's None if read fails
    else:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Markdown file not found at {markdown_path}. Skipping markdown mapping."
        )
        # Return the proposal as is, without markdown
        return initial_proposal

    if not markdown_content:
        # This case handles read failure or if file didn't exist and we decided to proceed earlier
        return initial_proposal

    console.print("Preparing request for markdown mapping...", style="dim")
    region_descriptions_for_prompt = []
    for i, item in enumerate(initial_proposal.items):
        region_descriptions_for_prompt.append(
            f"{i + 1}. Region ID: {item.id} \n   Description: {item.region_description}\n   Observed Content: {item.observed_content_summary}"
        )
    descriptions_text = "\n\n".join(region_descriptions_for_prompt)

    # Calculate length once for cleaner f-string
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

    console.print("Requesting markdown mapping...", style="dim")
    try:
        mapping_response = await client.beta.chat.completions.parse(
            model=MARKDOWN_MAPPING_MODEL,
            messages=[{"role": "user", "content": markdown_mapping_prompt}],
            response_format=MarkdownMappingResponse,
            max_tokens=MAX_TOKENS,
        )

        markdown_mapping = mapping_response.choices[0].message.parsed
        if (
            not markdown_mapping
            or markdown_mapping.mapped_items is None
            or len(markdown_mapping.mapped_items) != num_items  # Also check length
        ):
            console.print(
                "[bold red]Error:[/bold red] Markdown mapping call returned invalid content."
            )
            return initial_proposal  # Return original proposal

        # --- Step 3: Merge Results ---
        console.print("Merging markdown mapping results...", style="dim")
        for i, item in enumerate(markdown_mapping.mapped_items):
            # Use None if the snippet is empty or just whitespace, otherwise store the snippet
            initial_proposal.items[i].markdown_content = (
                item.markdown_snippet.strip()
                if item.markdown_snippet and item.markdown_snippet.strip()
                else None
            )
            # Store metadata if it's a non-empty dictionary, otherwise None
            initial_proposal.items[i].metadata = item.metadata if item.metadata else None

        console.print(
            "[bold green]Success:[/bold green] Merged markdown mapping results.", style="dim"
        )

        # --- Filter out items with no markdown or metadata ---
        if initial_proposal and initial_proposal.items:
            original_count = len(initial_proposal.items)
            initial_proposal.items = [
                item
                for item in initial_proposal.items
                if item.metadata is not None  # Keep only if metadata was found
            ]
            filtered_count = len(initial_proposal.items)
            if original_count != filtered_count:
                console.print(
                    f"Filtered out {original_count - filtered_count} items lacking markdown and metadata.",
                    style="dim",
                )

        return initial_proposal

    except openai.APIConnectionError as e:
        console.print(f"[bold red]Mapping Service Error:[/bold red] Failed to connect: {e}")
    except openai.RateLimitError as e:
        console.print(f"[bold red]Mapping Service Error:[/bold red] Rate limit exceeded: {e}")
    except openai.APIStatusError as e:
        console.print(
            f"[bold red]Mapping Service Error:[/bold red] Status {e.status_code}: {e.response}"
        )
    except Exception as e:
        # Catch potential validation errors from parse() or other issues
        console.print(f"[bold red]Error:[/bold red] Unexpected error during markdown mapping: {e}")

    # If markdown mapping failed for any reason, return the original proposal
    # Still apply filtering even if mapping failed, in case some items got populated before failure
    console.print(
        "[bold yellow]Warning:[/bold yellow] Markdown mapping failed. Returning proposal without full markdown/metadata."
    )
    # --- Filter out items with no markdown or metadata ---
    if initial_proposal and initial_proposal.items:
        original_count = len(initial_proposal.items)
        initial_proposal.items = [
            item
            for item in initial_proposal.items
            if item.metadata is not None  # Keep only if metadata was found
        ]
        filtered_count = len(initial_proposal.items)
        if original_count != filtered_count:
            console.print(
                f"Filtered out {original_count - filtered_count} items lacking markdown and metadata.",
                style="dim",
            )

    return initial_proposal


async def main():
    console.print("[bold blue]Starting Extraction Proposal Script...[/bold blue]")
    # Construct path from constants
    image_path = Path(f"samples/{SAMPLE_HOST}/{SAMPLE_SLUG}/{SAMPLE_SLUG}.jpg")

    if not image_path.exists():
        console.print(f"[bold red]Error:[/bold red] Image not found at {image_path}")
        return

    console.print(f"Target image: {image_path}")

    proposal = await propose_extractions(image_path)

    # Check if the proposal exists AND has items *after* internal filtering
    if proposal and proposal.items:
        console.print("\n[bold magenta]=== Proposed Extractions ===[/bold magenta]")
        # Convert Pydantic model to dict for pretty printing and saving
        proposal_dict = proposal.model_dump(mode="json")  # Use mode='json' for better serialization
        console.print(pretty_repr(proposal_dict, indent_size=2))

        # --- Save to JSON ---
        try:
            output_dir = image_path.parent
            output_path = output_dir / "proposal.json"
            # Ensure directory exists (though it should based on image existing)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(proposal_dict, f, indent=2, ensure_ascii=False)
            console.print(f"\n[bold green]Success:[/bold green] Proposal saved to {output_path}")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Failed to save proposal to JSON: {e}")

    elif proposal:  # Proposal exists but items list is empty after filtering
        console.print(
            "\n[bold yellow]Warning:[/bold yellow] No items with metadata found after processing. Proposal not printed or saved."
        )
    else:  # propose_extractions returned None (e.g., image analysis failed)
        console.print("\n[bold red]Failed to generate initial extraction proposal.[/bold red]")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY environment variable not set.")
    else:
        asyncio.run(main())

import asyncio
import base64
import os
from pathlib import Path
from typing import List, Optional

import openai
from pydantic import BaseModel, Field
from rich.console import Console
from rich.pretty import pretty_repr

# --- Model Constants ---
IMAGE_ANALYSIS_MODEL = "gpt-4.1"
MARKDOWN_MAPPING_MODEL = "gpt-4.1"

# --- Pydantic Models ---


class ProposedContentRegion(BaseModel):
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


class MarkdownMappingResponse(BaseModel):
    markdown_snippets: List[str] = Field(
        ...,
        description="A list of markdown text snippets. Each snippet corresponds to an item in the input list provided to the LLM, maintaining the original order.",
    )
    metadata_list: List[Optional[dict[str, str]]] = Field(
        ...,
        description="A list of dictionaries, each containing extracted metadata corresponding to an item in the input list. The order MUST be maintained. Use null or an empty dict if no metadata is found.",
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

    console.print("Initializing OpenAI client...", style="dim")
    try:
        # Using instructor requires patching the client
        client = openai.AsyncOpenAI()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to initialize OpenAI client: {e}")
        return None

    # --- Step 1: Image Analysis ---
    image_analysis_prompt = (
        "You are an expert analyst specializing in understanding webpage content from screenshots."
        "\nAnalyze the provided image, which is a screenshot of a webpage."
        "\nIdentify the major functional or informational regions visible."
        "\n**Important:** If you identify a region that contains a list or feed of similar items (like articles, products, posts), treat EACH item in the list as a SEPARATE region."
        "\nFor each region (or individual list item), provide:"
        "\n1. `region_description`: A clear description of the region or list item."
        "\n2. `observed_content_summary`: A brief summary of the actual text or content observed within that region/item in the image."
        "\nDo NOT suggest CSS selectors or HTML element names. Focus ONLY on the visible content and structure."
        "\nRespond using the structure defined by the provided Pydantic model."
    )

    console.print(
        "Sending request to OpenAI for image analysis... (This might take a moment)", style="dim"
    )
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
            max_tokens=1024,
        )

        initial_proposal = response.choices[0].message.parsed
        if not initial_proposal:
            console.print(
                "[bold red]Error:[/bold red] OpenAI returned empty parsed content for image analysis."
            )
            # Decide if we should return None or continue without markdown
            return None  # Let's return None if the primary step fails

        console.print(
            "[bold green]Success:[/bold green] Received and parsed initial proposal from OpenAI."
        )

    except openai.APIConnectionError as e:
        console.print(f"[bold red]OpenAI Error (Image Analysis):[/bold red] Failed to connect: {e}")
        return None
    except openai.RateLimitError as e:
        console.print(
            f"[bold red]OpenAI Error (Image Analysis):[/bold red] Rate limit exceeded: {e}"
        )
        return None
    except openai.APIStatusError as e:
        console.print(
            f"[bold red]OpenAI Error (Image Analysis):[/bold red] Status {e.status_code}: {e.response}"
        )
        return None
    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] An unexpected error occurred during image analysis API call: {e}"
        )
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

    console.print("Preparing second request to map markdown content...", style="dim")
    region_descriptions_for_prompt = []
    for i, item in enumerate(initial_proposal.items):
        region_descriptions_for_prompt.append(
            f"{i + 1}. Description: {item.region_description}\n   Observed Content: {item.observed_content_summary}"
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
*   `markdown_snippets`: A list containing exactly {num_items} strings. Each string is the corresponding markdown snippet. Use an empty string "" if no *direct* match is found.
*   `metadata_list`: A list containing exactly {num_items} dictionaries (or null). Each dictionary holds the factual, discrete key-value metadata extracted. **Use an empty dictionary {{}} or null if NO verifiable metadata can be extracted for a region.**

Maintain the exact order from the input regions in both lists. Prioritize accuracy and avoid hallucination; only report what is demonstrably present."""

    console.print("Sending second request to OpenAI for markdown mapping...", style="dim")
    try:
        # Assuming client is still valid; re-init if necessary or handle potential closure
        mapping_response = await client.beta.chat.completions.parse(
            model=MARKDOWN_MAPPING_MODEL,
            messages=[{"role": "user", "content": markdown_mapping_prompt}],
            response_format=MarkdownMappingResponse,
            max_tokens=2048,  # Increased token limit for markdown context
        )

        markdown_mapping = mapping_response.choices[0].message.parsed
        if (
            not markdown_mapping
            or markdown_mapping.markdown_snippets is None
            or markdown_mapping.metadata_list is None  # Also check metadata_list
        ):
            console.print(
                "[bold red]Error:[/bold red] Markdown mapping call returned invalid content."
            )
            return initial_proposal  # Return original proposal

        # --- Step 3: Merge Results ---
        if (
            len(markdown_mapping.markdown_snippets) != num_items
            or len(markdown_mapping.metadata_list) != num_items  # Check metadata length too
        ):
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Mismatch between number of proposed items ({num_items}) and returned markdown/metadata snippets ({len(markdown_mapping.markdown_snippets)}/{len(markdown_mapping.metadata_list)}). Skipping merge."
            )
            return initial_proposal

        console.print("Merging markdown snippets and metadata into proposal...", style="dim")
        for i, (snippet, meta) in enumerate(
            zip(markdown_mapping.markdown_snippets, markdown_mapping.metadata_list, strict=False)
        ):
            # Use None if the snippet is empty or just whitespace, otherwise store the snippet
            initial_proposal.items[i].markdown_content = (
                snippet.strip() if snippet and snippet.strip() else None
            )
            # Store metadata if it's a non-empty dictionary, otherwise None
            initial_proposal.items[i].metadata = meta if meta else None

        console.print(
            "[bold green]Success:[/bold green] Successfully mapped and merged markdown content and metadata."
        )
        return initial_proposal

    except openai.APIConnectionError as e:
        console.print(
            f"[bold red]OpenAI Error (Markdown Mapping):[/bold red] Failed to connect: {e}"
        )
    except openai.RateLimitError as e:
        console.print(
            f"[bold red]OpenAI Error (Markdown Mapping):[/bold red] Rate limit exceeded: {e}"
        )
    except openai.APIStatusError as e:
        console.print(
            f"[bold red]OpenAI Error (Markdown Mapping):[/bold red] Status {e.status_code}: {e.response}"
        )
    except Exception as e:
        # Catch potential validation errors from parse() or other issues
        console.print(
            f"[bold red]Error:[/bold red] An unexpected error occurred during markdown mapping: {e}"
        )

    # If markdown mapping failed for any reason, return the original proposal
    console.print(
        "[bold yellow]Warning:[/bold yellow] Markdown mapping failed. Returning proposal without markdown content."
    )
    return initial_proposal


async def main():
    console.print("[bold blue]Starting Extraction Proposal Script...[/bold blue]")
    image_path = Path("samples/substack.com/home/home.jpg")

    if not image_path.exists():
        console.print(f"[bold red]Error:[/bold red] Image not found at {image_path}")
        return

    console.print(f"Target image: {image_path}")

    proposal = await propose_extractions(image_path)

    if proposal:
        console.print("\n[bold magenta]=== Proposed Extractions ===[/bold magenta]")
        console.print(pretty_repr(proposal.model_dump(), indent_size=2))
    else:
        console.print("\n[bold red]Failed to generate extraction proposal.[/bold red]")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY environment variable not set.")
    else:
        asyncio.run(main())

import asyncio
import base64
import os
from pathlib import Path
from typing import List, Optional

import openai
from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel, Field
from rich.console import Console
from rich.pretty import pretty_repr

# --- Model Constants ---
IMAGE_ANALYSIS_MODEL = "gpt-4.1-nano"
MARKDOWN_MAPPING_MODEL = "gpt-4.1"
SELECTOR_MAPPING_MODEL = "gpt-4.1-nano"

MAX_TOKENS = 4096

# --- Pydantic Models ---


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
    css_selector: Optional[str] = Field(
        None,
        description="The CSS selector for the main container element of this region, if found.",
    )


class ExtractionProposal(BaseModel):
    items: List[ProposedContentRegion] = Field(
        ..., description="List of proposed content regions identified in the webpage screenshot."
    )


class MarkdownMappingItem(BaseModel):
    markdown_snippet: Optional[str] = Field(
        None, description="The corresponding markdown snippet for the region, or null if none found."
    )
    metadata: Optional[dict[str, str]] = Field(
        None, description="Extracted factual metadata dictionary, or null/empty if none found."
    )


class MarkdownMappingResponse(BaseModel):
    mapped_items: List[MarkdownMappingItem] = Field(
        ...,
        description="A list containing exactly one item for each input region, in the original order. Each item bundles the markdown snippet and metadata for that region.",
    )


class SelectorMappingResponse(BaseModel):
    css_selectors: List[Optional[str]] = Field(
        ...,
        description="A list of CSS selector strings. Each selector corresponds to an item in the input list, maintaining the original order. Use null if no suitable selector is found.",
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
            max_tokens=MAX_TOKENS,
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

        # --- Assign sequential IDs --- 
        if initial_proposal and initial_proposal.items:
            for i, item in enumerate(initial_proposal.items):
                item.id = i + 1 # Assign 1-based IDs
            console.print(f"Assigned IDs to {len(initial_proposal.items)} proposed regions.", style="dim")

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

    console.print("Sending second request to OpenAI for markdown mapping...", style="dim")
    try:
        # Assuming client is still valid; re-init if necessary or handle potential closure
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
        console.print("Merging markdown snippets and metadata into proposal...", style="dim")
        for i, item in enumerate(markdown_mapping.mapped_items):
            # Use None if the snippet is empty or just whitespace, otherwise store the snippet
            initial_proposal.items[i].markdown_content = (
                item.markdown_snippet.strip() if item.markdown_snippet and item.markdown_snippet.strip() else None
            )
            # Store metadata if it's a non-empty dictionary, otherwise None
            initial_proposal.items[i].metadata = item.metadata if item.metadata else None

        console.print(
            "[bold green]Success:[/bold green] Successfully mapped and merged markdown content and metadata."
        )

        # --- Step 3: HTML Selector Mapping ---
        html_path = image_path.with_suffix(".html")
        html_content = None
        if html_path.exists():
            console.print(f"Attempting to read corresponding HTML file: {html_path}", style="dim")
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                console.print("Successfully read HTML file.", style="dim")
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Failed to read HTML file {html_path}: {e}. Proceeding without CSS selectors."
                )
                html_content = None
        else:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] HTML file not found at {html_path}. Skipping selector mapping."
            )
            # Return the proposal as enriched by step 2
            return initial_proposal

        if not html_content:
            return initial_proposal

        # --- Clean HTML before sending ---
        console.print("Cleaning HTML content...", style="dim")
        try:
            soup = BeautifulSoup(html_content, "lxml")
            # Remove script and style tags
            for tag in soup(["script", "style", "svg"]):  # Also removing SVG for now
                tag.decompose()
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            # Get cleaned HTML (consider .prettify() if whitespace is needed, but stripped is smaller)
            cleaned_html_content = str(soup)
            original_len = len(html_content)
            cleaned_len = len(cleaned_html_content)
            console.print(
                f"HTML reduced from {original_len} to {cleaned_len} characters ({100 * cleaned_len / original_len:.1f}%).",
                style="dim",
            )
        except Exception as e:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Failed to clean HTML: {e}. Using original HTML."
            )
            cleaned_html_content = html_content  # Fallback to original if cleaning fails

        # --- Step 3: Chunked CSS Selector Mapping ---
        console.print("Starting chunked CSS selector mapping...", style="dim")

        # Settings for chunking
        # Estimate based on ~3-4 chars/token. Target < 30k tokens for GPT-4.1 TPM limit.
        # Let's aim for ~100k chars per chunk + prompt buffer.
        TARGET_CHUNK_SIZE_CHARS = 100000
        html_chunks = [
            cleaned_html_content[i : i + TARGET_CHUNK_SIZE_CHARS]
            for i in range(0, len(cleaned_html_content), TARGET_CHUNK_SIZE_CHARS)
        ]
        console.print(f"Split cleaned HTML into {len(html_chunks)} chunks.", style="dim")

        # Track which item indices still need selectors
        remaining_indices = set(range(num_items))
        found_selectors: dict[int, str] = {}

        for chunk_index, html_chunk in enumerate(html_chunks):
            if not remaining_indices:
                console.print("All selectors found, stopping chunk iteration.", style="dim")
                break

            current_indices_list = sorted(remaining_indices)
            console.print(
                f"  Processing chunk {chunk_index + 1}/{len(html_chunks)} for {len(current_indices_list)} remaining items...",
                style="dim",
            )

            # Prepare descriptions for the items we're looking for in *this* chunk
            region_descriptions_for_chunk_prompt = []
            original_index_map = {}
            for i, original_index in enumerate(current_indices_list):
                item = initial_proposal.items[original_index]
                region_descriptions_for_chunk_prompt.append(
                    f"{i + 1}. (Region ID: {item.id}) Description: {item.region_description}\n   Observed Content: {item.observed_content_summary}"
                )
                original_index_map[i] = (
                    original_index  # Map response index back to original item index
                )

            descriptions_text_chunk = "\n\n".join(region_descriptions_for_chunk_prompt)
            num_regions_in_chunk = len(current_indices_list)

            # Construct the prompt for this chunk
            selector_mapping_prompt = f"""You are an expert web developer tasked with finding robust CSS selectors.

The *current chunk* of the webpage's HTML content is below:
--- HTML CHUNK START ---
{html_chunk}
--- HTML CHUNK END ---

Your task is to find selectors *only for the following {num_regions_in_chunk} regions* IF their main container element appears to be primarily located *within this specific HTML chunk*:
--- REGIONS TO FIND IN THIS CHUNK ({num_regions_in_chunk} total) ---
{descriptions_text_chunk}
--- END REGIONS ---

Instructions for EACH region above:
1.  Determine if its main container element is likely present in the provided HTML CHUNK.
2.  If present, provide the *single best CSS selector* for that region's container.
3.  **Selector Quality Guidelines (MUST Follow Strictly):**
    *   **Prioritize:** Unique IDs (`#element-id`), stable `data-*` attributes (`[data-testid="..."]`), or specific, meaningful class combinations (`.meaningful-class.specific-context`). These are robust against page changes.
    *   **AVOID:** Returning only generic tag names (`div`, `span`, `a`, `p`). These are useless.
    *   **AVOID:** Highly brittle positional selectors (`:nth-child`, `:nth-of-type`) as they break easily. Use ONLY as a last resort if items truly lack unique identifiers.
    *   **AVOID:** Relying solely on text content (`:contains(...)` - non-standard and unreliable) if structural selectors exist.
    *   The selector MUST be specific enough to **uniquely target** the intended container for the described region within the broader page context, not just within the chunk.
    *   **Goal:** Find the most stable, robust, and specific selector possible.
4.  If a region's main content does NOT appear to be in this chunk, or you cannot find a selector meeting these strict quality guidelines, use `null` for that region.

Respond with a JSON object adhering precisely to the `SelectorMappingResponse` schema. The object must contain one key:
*   `css_selectors`: A list containing exactly {num_regions_in_chunk} strings or nulls. The order MUST correspond EXACTLY to the order of the regions listed above (1 to {num_regions_in_chunk}).

Provide ONLY the selector strings or nulls in the list, no explanations.
"""

            try:
                # Use the original SelectorMappingResponse, but expect length == num_regions_in_chunk
                selector_response = await client.beta.chat.completions.parse(
                    model=SELECTOR_MAPPING_MODEL,
                    messages=[{"role": "user", "content": selector_mapping_prompt}],
                    response_format=SelectorMappingResponse,
                    max_tokens=MAX_TOKENS,
                )

                chunk_mapping = selector_response.choices[0].message.parsed
                if not chunk_mapping or chunk_mapping.css_selectors is None:
                    console.print(
                        f"    [bold yellow]Warning:[/bold yellow] Received invalid content for chunk {chunk_index + 1}. Skipping chunk."
                    )
                    continue

                if len(chunk_mapping.css_selectors) != num_regions_in_chunk:
                    console.print(
                        f"    [bold yellow]Warning:[/bold yellow] Mismatch in expected ({num_regions_in_chunk}) vs received ({len(chunk_mapping.css_selectors)}) selectors for chunk {chunk_index + 1}. Skipping chunk."
                    )
                    continue

                # Process results for this chunk
                newly_found_count = 0
                for i, selector in enumerate(chunk_mapping.css_selectors):
                    original_item_index = original_index_map.get(i)
                    if original_item_index is None:
                        continue  # Should not happen

                    # Only process if we still need this index
                    if original_item_index in remaining_indices:
                        is_valid_selector = False
                        if selector and selector.strip() and selector.lower() != "null":
                            selector = selector.strip()
                            # --- Basic Validation ---
                            try:
                                # Check 1: Basic syntax and existence check
                                found_element = soup.select_one(selector)
                                if found_element:
                                    # Check 2: Reject overly generic tag-only selectors
                                    simple_tags = {'div', 'span', 'a', 'p', 'ul', 'li', 'h1', 'h2', 'h3', 'img'}
                                    if selector in simple_tags:
                                        console.print(f"      Selector '{selector}' for item {original_item_index + 1} is too generic (tag only). Invalid.", style="yellow")
                                        is_valid_selector = False
                                    else:
                                        is_valid_selector = True # Passed both checks
                            except Exception as e:
                                # Catch potential errors from invalid selector syntax
                                console.print(
                                    f"      Selector '{selector}' for item {original_item_index + 1} caused validation error: {e}. Invalid.",
                                    style="red",
                                )
                        # --- End Validation ---

                    if is_valid_selector:
                        # Explicit check to satisfy type checker
                        if selector is not None:
                            found_selectors[original_item_index] = selector
                            initial_proposal.items[original_item_index].css_selector = selector
                            remaining_indices.remove(original_item_index)
                            newly_found_count += 1
                            console.print(
                                f"      Found and validated selector for original item {original_item_index + 1}: {selector}",
                                style="cyan",
                            )
                    # else: If selector was null or invalid, do nothing, leave item in remaining_indices

                console.print(
                    f"    Processed chunk {chunk_index + 1}. Found {newly_found_count} new valid selectors.",
                    style="dim",
                )

            except openai.APIConnectionError as e:
                console.print(
                    f"    [bold red]Error (Chunk {chunk_index + 1}):[/bold red] Failed to connect: {e}"
                )
            except openai.RateLimitError as e:
                console.print(
                    f"    [bold red]Error (Chunk {chunk_index + 1}):[/bold red] Rate limit exceeded: {e}"
                )
                console.print("    Slowing down... adding extra delay.")
                await asyncio.sleep(10)  # Longer delay if rate limited
            except openai.APIStatusError as e:
                console.print(
                    f"    [bold red]Error (Chunk {chunk_index + 1}):[/bold red] Status {e.status_code}: {e.response}"
                )
            except Exception as e:
                console.print(
                    f"    [bold red]Error (Chunk {chunk_index + 1}):[/bold red] Unexpected error: {e}"
                )

            # Add a delay between chunk requests
            await asyncio.sleep(2)  # Adjust delay as needed

        console.print(
            f"Finished chunked mapping. Found selectors for {len(found_selectors)}/{num_items} items.",
            style="dim",
        )
        if remaining_indices:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Could not find selectors for items (original indices): {sorted(remaining_indices)}",
                style="yellow",
            )

        # Assign null to any remaining items explicitly (though they should be None by default)
        for index in remaining_indices:
            initial_proposal.items[index].css_selector = None

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

import asyncio
import base64
import os
from pathlib import Path
from typing import List, Optional

import openai
from pydantic import BaseModel, Field
from rich.console import Console
from rich.pretty import pretty_repr

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


class ExtractionProposal(BaseModel):
    items: List[ProposedContentRegion] = Field(
        ..., description="List of proposed content regions identified in the webpage screenshot."
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

    prompt = (
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

    console.print("Sending request to OpenAI... (This might take a moment)", style="dim")
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4.1-mini",  # Or "gpt-4o"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },  # Use high detail for better analysis
                        },
                    ],
                }
            ],
            response_format=ExtractionProposal,  # Pass the Pydantic model directly
            max_tokens=1024,  # Adjust as needed
        )

        # Access the parsed Pydantic object directly
        proposal = response.choices[0].message.parsed
        if not proposal:
            console.print("[bold red]Error:[/bold red] OpenAI returned empty parsed content.")
            return None

        console.print("[bold green]Success:[/bold green] Received and parsed proposal from OpenAI.")
        return proposal

    except openai.APIConnectionError as e:
        console.print(f"[bold red]OpenAI Error:[/bold red] Failed to connect: {e}")
    except openai.RateLimitError as e:
        console.print(f"[bold red]OpenAI Error:[/bold red] Rate limit exceeded: {e}")
    except openai.APIStatusError as e:
        console.print(f"[bold red]OpenAI Error:[/bold red] Status {e.status_code}: {e.response}")
    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] An unexpected error occurred during OpenAI API call: {e}"
        )

    return None


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

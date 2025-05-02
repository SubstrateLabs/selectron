import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import openai
from pydantic import BaseModel, Field
from rich import print
from rich.pretty import pretty_repr

from selectron.ai.analyze_types import ExtractionProposal
from selectron.ai.selector_agent import AgentResult, SelectorAgent
from selectron.util.logger import get_logger

logger = get_logger(__name__)

# --- Configuration ---
ENRICHMENT_MODEL = "gpt-4.1-mini"  # Cheaper model for enrichment
MAX_DOM_CONTEXT_LEN = 8000  # Limit DOM context for enrichment prompt

# Use the linkedin sample for this implementation
SAMPLE_HOST = "www.linkedin.com"
SAMPLE_SLUG = "feed"

sample_path = Path(f"samples/{SAMPLE_HOST}/{SAMPLE_SLUG}/{SAMPLE_SLUG}")
sample_dir = sample_path.parent
sample_html_path = sample_path.with_suffix(".html")
# sample_jpg_path = sample_path.with_suffix(".jpg") # No longer needed directly
# sample_md_path = sample_path.with_suffix(".md") # No longer needed directly
# sample_dom_path = sample_path.with_suffix(".dom.txt") # No longer needed directly
sample_proposal_path = sample_dir / "proposal.json"
BASE_URL = f"https://{SAMPLE_HOST}/"  # Derive base URL


# Define the AgentRequest structure
@dataclass
class AgentRequest:
    target_description: str
    verification_text: Optional[str]
    attribute_to_extract: Optional[str]
    extract_text: bool
    request_type: Literal["region_selection", "metadata_extraction"]
    region_id: int
    metadata_key: Optional[str] = None
    proposed_selector_hint: Optional[str] = None


# Define the structure for the enrichment LLM call
class ExtractionGoal(BaseModel):
    attribute_to_extract: Optional[str] = Field(
        None,
        description="The specific HTML attribute to extract (e.g., 'href', 'src') if appropriate, otherwise null.",
    )
    extract_text: bool = Field(..., description="Whether to extract the element's text content.")
    updated_target_description: Optional[str] = Field(
        None,
        description="An improved target_description for the SelectorAgent, if the original can be refined based on context/DOM. Otherwise null.",
    )
    updated_verification_text: Optional[str] = Field(
        None,
        description="An improved verification_text for the SelectorAgent, if the original metadata value can be improved using context/DOM. Otherwise null.",
    )
    proposed_selector: Optional[str] = Field(
        None,
        description="A plausible CSS selector hint targeting the element, based on context and simplified DOM. May require refinement."
    )
    reasoning: str = Field(
        ..., description="Brief explanation for the chosen extraction method and any updates made."
    )


async def determine_extraction_goal(
    client: openai.AsyncOpenAI,
    metadata_key: str,
    metadata_value: str,
    region_description: str,
    dom_context: Optional[str],
) -> Optional[ExtractionGoal]:
    """Uses an LLM to refine the extraction goal for a specific metadata item."""

    # Basic prompt components
    prompt_parts = [
        "You are an expert assistant refining extraction goals for a web scraping agent.",
        f"The target region is described as: '{region_description}'",
        f"Within this region, we want to extract the data associated with the key '{metadata_key}', which has the value '{metadata_value}'.",
        "Based on the key, value, and overall region context, determine the best way to extract this specific data point from its corresponding HTML element.",
        "Consider if the data is likely found in an element's attribute (like 'href' for links, 'src' for images, 'alt', 'title') or within its text content.",
        "Default Target Description: \"The specific element representing '{key}' (value: '{value}') within the region for '{region_description}'\"",
        f'Default Verification Text: "{metadata_value}"',
        "Can you refine the Default Target Description or Verification Text for clarity or robustness, potentially using the simplified DOM context if provided?",
        "Also, based on the context and especially the simplified DOM, propose a plausible CSS selector hint (`proposed_selector`) that might target this specific metadata element. This is just a hint and may need refinement."
    ]

    # Add DOM context if available
    if dom_context:
        truncated_dom = dom_context[:MAX_DOM_CONTEXT_LEN]
        if len(dom_context) > MAX_DOM_CONTEXT_LEN:
            truncated_dom += "\n... (DOM truncated)"
        prompt_parts.extend(
            [
                "\nSimplified DOM Context (format: [id]<tag attrs...> text):",
                "--- DOM START ---",
                truncated_dom,
                "--- DOM END ---",
            ]
        )
    else:
        prompt_parts.append("No simplified DOM context provided.")

    prompt_parts.append("Respond ONLY with a JSON object matching the ExtractionGoal schema.")
    prompt = "\n".join(prompt_parts)

    try:
        response = await client.beta.chat.completions.parse(
            model=ENRICHMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractionGoal,
            # max_tokens might be needed if descriptions/values are long
        )
        goal = response.choices[0].message.parsed
        if goal and goal.reasoning:  # Check goal exists before accessing reasoning
            logger.debug(f"Enrichment successful for key '{metadata_key}': {goal.reasoning}")
        else:
            logger.debug(
                f"Enrichment successful for key '{metadata_key}' (no specific reasoning provided)."
            )
        return goal
    except Exception as e:
        logger.error(f"LLM enrichment failed for key '{metadata_key}': {e}", exc_info=True)
        return None


async def create_agent_requests_from_proposal(
    proposal: ExtractionProposal, client: openai.AsyncOpenAI, dom_content: Optional[str]
) -> list[AgentRequest]:
    """Generates a list of AgentRequest objects from a proposal, enriching metadata requests via LLM."""
    agent_requests: list[AgentRequest] = []

    for item in proposal.items:
        if item.id is None:
            logger.warning(f"Skipping proposal item without ID: {item.region_description}")
            continue

        # Filter: Initially require metadata for simplicity, might relax later
        if item.metadata is None:
            logger.info(f"Skipping region {item.id} due to missing metadata.")
            continue

        # Generate region-level request (selector finder)
        summary_snippet = (
            item.observed_content_summary[:100] + "..."
            if len(item.observed_content_summary) > 100
            else item.observed_content_summary
        )
        region_desc = f"The container element for the region described as '{item.region_description}' which contains content like '{summary_snippet}'"

        # Use the longest metadata string value for verification if available, otherwise None
        region_verification: Optional[str] = None
        if item.metadata:
            string_values = [
                str(v)
                for v in item.metadata.values()
                if isinstance(v, (str, int, float)) and str(v).strip()
            ]
            if string_values:
                longest_value = max(string_values, key=len)
                region_verification = longest_value[:100] + (
                    "..." if len(longest_value) > 100 else ""
                )  # Corrected parenthesis

        agent_requests.append(
            AgentRequest(
                target_description=region_desc,
                verification_text=region_verification,
                attribute_to_extract=None,
                extract_text=False,
                request_type="region_selection",
                region_id=item.id,
                metadata_key=None,
                proposed_selector_hint=None,
            )
        )

        # Generate metadata-level requests (data extractors)
        for key, value in item.metadata.items():
            if not isinstance(value, str) or not value.strip():
                logger.debug(
                    f"Skipping metadata key '{key}' for region {item.id} due to non-string or empty value."
                )
                continue

            # --- LLM Enrichment Step ---
            logger.debug(f"Attempting LLM enrichment for Region {item.id}, Key '{key}'")
            goal: Optional[ExtractionGoal] = await determine_extraction_goal(
                client=client,
                metadata_key=key,
                metadata_value=value,
                region_description=item.region_description,
                dom_context=dom_content,
            )

            # --- Determine final request parameters ---
            target_description: str
            verification_text: Optional[str]
            attribute_to_extract: Optional[str] = None
            extract_text: bool = True  # Default if enrichment fails
            proposed_selector_hint: Optional[str] = None

            # Use enrichment results if successful
            if goal:
                attribute_to_extract = goal.attribute_to_extract
                extract_text = goal.extract_text
                target_description = (
                    goal.updated_target_description
                    or f"The specific element representing '{key}' (value: '{value[:100]}...') within the region for '{item.region_description}'"
                )  # Fallback desc
                verification_text = goal.updated_verification_text or value[:150] + (
                    "..." if len(value) > 150 else ""
                )  # Fallback verification
                proposed_selector_hint = goal.proposed_selector
            else:
                # Fallback if enrichment failed
                logger.warning(
                    f"Using fallback for Region {item.id}, Key '{key}' due to enrichment failure."
                )
                target_description = f"The specific element representing '{key}' (value: '{value[:100]}...') within the region for '{item.region_description}'"
                verification_text = value[:150] + ("..." if len(value) > 150 else "")

            # Ensure verification text isn't excessively long even after potential update
            if verification_text and len(verification_text) > 150:
                verification_text = verification_text[:150] + "..."

            agent_requests.append(
                AgentRequest(
                    target_description=target_description,
                    verification_text=verification_text,
                    attribute_to_extract=attribute_to_extract,
                    extract_text=extract_text,
                    request_type="metadata_extraction",
                    region_id=item.id,
                    metadata_key=key,
                    proposed_selector_hint=proposed_selector_hint,
                )
            )

    logger.info(f"Generated {len(agent_requests)} agent requests from proposal.")
    return agent_requests


async def main():
    # Ensure OPENAI_API_KEY is set early
    import os

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
        return

    # --- Instantiate OpenAI Client ---
    try:
        client = openai.AsyncOpenAI()
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        return

    # --- Load Inputs ---
    print(f"📂 Loading inputs from: {sample_dir}")

    if not sample_proposal_path.exists():
        print(f"[bold red]Error:[/bold red] Proposal file not found: {sample_proposal_path}")
        return

    if not sample_html_path.exists():
        print(f"[bold red]Error:[/bold red] HTML file not found: {sample_html_path}")
        return

    try:
        with open(sample_proposal_path, "r", encoding="utf-8") as f:
            proposal_data = json.load(f)
            # Use cast to inform type checker, validation happens implicitly
            proposal = ExtractionProposal(**proposal_data)
        print(f"✅ Loaded proposal: {sample_proposal_path}")
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to load or parse proposal JSON: {e}")
        return

    try:
        with open(sample_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        print(f"✅ Loaded HTML: {sample_html_path}")
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to load HTML file: {e}")
        return

    # --- Load Optional DOM Representation ---
    dom_content: Optional[str] = None
    sample_dom_path = sample_path.with_suffix(".dom.txt")  # Define path here
    if sample_dom_path.exists():
        try:
            with open(sample_dom_path, "r", encoding="utf-8") as f:
                dom_content = f.read()
            print(f"✅ Loaded optional DOM representation: {sample_dom_path}")
        except Exception as e:
            print(
                f"[bold yellow]Warning:[/bold yellow] Failed to load DOM file {sample_dom_path}: {e}. Proceeding without it."
            )
            dom_content = None
    else:
        print(
            f"[bold dim]Info:[/bold dim] Optional DOM file not found: {sample_dom_path}. Proceeding without it."
        )

    # --- Generate Agent Requests ---
    print("\n🧠 Generating agent requests from proposal (with enrichment)...")
    agent_requests = await create_agent_requests_from_proposal(
        proposal=proposal, client=client, dom_content=dom_content
    )

    if not agent_requests:
        print("[bold yellow]Warning:[/bold yellow] No agent requests generated from the proposal.")
        return

    print(f"✅ Generated {len(agent_requests)} requests.")
    # Optionally print requests for debugging:
    # for req in agent_requests:
    #     print(pretty_repr(req, indent_size=2))

    # --- Verify Agent Requests (Early Return) ---
    print("\n📝 Verifying generated agent requests...")

    # Separate requests by type
    region_requests = [req for req in agent_requests if req.request_type == "region_selection"]
    metadata_requests = [req for req in agent_requests if req.request_type == "metadata_extraction"]

    if region_requests:
        print("\n--- Region Selection Requests ---")
        for i, request in enumerate(region_requests):
            print(
                f"\n--- Region Request {i + 1}/{len(region_requests)} (Overall {agent_requests.index(request) + 1}/{len(agent_requests)}) ---"
            )
            print(pretty_repr(request, indent_size=2))

    if metadata_requests:
        print("\n--- Metadata Extraction Requests ---")
        for i, request in enumerate(metadata_requests):
            print(
                f"\n--- Metadata Request {i + 1}/{len(metadata_requests)} (Overall {agent_requests.index(request) + 1}/{len(agent_requests)}) ---"
            )
            # Print hint if available
            if request.proposed_selector_hint:
                print(f"  Proposed Selector Hint: {request.proposed_selector_hint}")
            print(pretty_repr(request, indent_size=2))

    print("\n✅ Verification complete. Exiting script before agent execution.")
    return  # Early return for verification

    # --- Instantiate Agent ---
    print(f"\n🤖 Instantiating SelectorAgent (Base URL: {BASE_URL})...")
    try:
        agent_instance = SelectorAgent(
            html_content=html_content,
            base_url=BASE_URL,
            dom_representation=dom_content,  # Pass optional DOM content
        )
    except ValueError as ve:
        print(f"[bold red]Initialization Error:[/bold red] {ve}")
        return
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to instantiate SelectorAgent: {e}")
        return

    # --- Run Agent Requests ---
    print("🚀 Running agent requests...")
    results_store: dict[tuple[int, Optional[str]], AgentResult] = {}

    for i, request in enumerate(agent_requests):
        print(
            f"--- Running Request {i + 1}/{len(agent_requests)} ({request.request_type}: Region {request.region_id}"
            f"{f', Key: {request.metadata_key}' if request.metadata_key else ''}) ---"
        )
        print(f"  Description: {request.target_description[:100]}...")  # Truncate desc
        print(f"  Verification: {request.verification_text}")
        print(
            f"  Extract Attr: {request.attribute_to_extract}, Extract Text: {request.extract_text}"
        )

        try:
            result: AgentResult = await agent_instance.find_and_extract(
                target_description=request.target_description,
                attribute_to_extract=request.attribute_to_extract,
                extract_text=request.extract_text,
                verification_text=request.verification_text,
            )

            results_store[(request.region_id, request.metadata_key)] = result

            # --- Print Individual Result ---
            print(f"  ✅ Agent finished request {i + 1}.")
            print(f"  Proposed Selector: {result.proposed_selector}")
            print(f"  Reasoning: {result.reasoning[:100]}...")  # Truncate reasoning

            eval_result = result.final_verification
            extraction_successful = False
            if result.extraction_result.error:
                print(f"  Extraction Error: {result.extraction_result.error}")
            else:
                if result.attribute_extracted:
                    extraction_successful = (
                        result.extraction_result.extracted_attribute_value is not None
                    )
                    print(
                        f"  Extracted '{result.attribute_extracted}': {result.extraction_result.extracted_attribute_value}"
                    )
                if result.text_extracted_flag:
                    extraction_successful = result.extraction_result.extracted_text is not None
                    print(f"  Extracted Text: {result.extraction_result.extracted_text}")
                if not result.attribute_extracted and not result.text_extracted_flag:
                    extraction_successful = True  # No extraction requested, counts as success

            verification_successful = eval_result.element_count == 1 and not eval_result.error
            print(
                f"  Verification: Count={eval_result.element_count}, Error='{eval_result.error}', FoundText={eval_result.target_text_found_in_any_match}"
            )

            if extraction_successful and verification_successful:
                print("  [bold green]Status: SUCCESS[/bold green]")
            elif verification_successful:
                print(
                    "  [bold yellow]Status: PARTIAL (Verification OK, Extraction Failed/Empty)[/bold yellow]"
                )
            else:
                print("  [bold red]Status: FAILED (Verification Failed)[/bold red]")

        except Exception as e:
            print(
                f"[bold red]Error:[/bold red] Unhandled exception during agent run for request {i + 1}: {e}"
            )
            # Store a placeholder or skip? For now, just print and continue.

    # --- Final Summary (Optional) ---
    print("🏁 All agent requests processed.")
    successful_requests = sum(
        1
        for res in results_store.values()
        if res.final_verification.element_count == 1
        and not res.final_verification.error
        and not res.extraction_result.error  # Simplified success check
    )
    print(f"Total Requests: {len(agent_requests)}, Successful Verifications: {successful_requests}")
    # TODO: Add more sophisticated aggregation/output based on results_store


if __name__ == "__main__":
    asyncio.run(main())

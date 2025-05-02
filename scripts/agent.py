import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from rich import print
from rich.markdown import Markdown

# from rich.syntax import Syntax
from selectron.ai.analyze_types import ExtractionProposal
from selectron.ai.selector_agent import AgentResult, SelectorAgent
from selectron.util.logger import get_logger

logger = get_logger(__name__)

# --- Configuration ---
ENRICHMENT_MODEL = "gpt-4.1"
MAX_DOM_CONTEXT_LEN = 8000
AGENT_CONCURRENCY = 3

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
        description="A plausible CSS selector hint targeting the element, based on context and simplified DOM. May require refinement.",
    )
    reasoning: str = Field(
        ..., description="Brief explanation for the chosen extraction method and any updates made."
    )


# Modify create_and_start_agent_tasks
async def create_and_start_agent_tasks(
    proposal: ExtractionProposal,
    agent_instance: SelectorAgent,
) -> list[tuple[asyncio.Task, AgentRequest]]:
    """Generates agent requests for regions ONLY and starts agent tasks concurrently."""
    agent_tasks: list[tuple[asyncio.Task, AgentRequest]] = []
    logger.info("Starting agent task creation for REGION SELECTION requests only...")

    for item in proposal.items:
        if item.id is None:
            logger.warning(f"Skipping proposal item without ID: {item.region_description}")
            continue

        # Filter: Skip items without metadata for region selection? Or proceed?
        # Let's proceed even without metadata for region selection for now.
        # if item.metadata is None:
        #     logger.info(f"Skipping region {item.id} due to missing metadata.")
        #     continue

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
                )

        region_request = AgentRequest(
            target_description=region_desc,
            verification_text=region_verification,
            attribute_to_extract=None,
            extract_text=False,
            request_type="region_selection",
            region_id=item.id,
            metadata_key=None,
            proposed_selector_hint=None,  # No hint generation for regions yet
        )

        # Start the agent task for the region selection request
        region_task = asyncio.create_task(
            agent_instance.find_and_extract(
                target_description=region_request.target_description,
                attribute_to_extract=region_request.attribute_to_extract,
                extract_text=region_request.extract_text,
                verification_text=region_request.verification_text,
            )
        )
        agent_tasks.append((region_task, region_request))
        logger.debug(f"Started agent task for Region {item.id} selection.")

    logger.info(f"Created and started {len(agent_tasks)} REGION SELECTION agent tasks.")
    return agent_tasks


async def main():
    # Ensure OPENAI_API_KEY is set early
    import os

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
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

    # --- Instantiate Agent ---
    print(f"\n🤖 Instantiating SelectorAgent (Base URL: {BASE_URL})...")
    # Create semaphore here and pass directly to agent
    agent_semaphore = asyncio.Semaphore(AGENT_CONCURRENCY)
    logger.info(f"Agent concurrency limited to {AGENT_CONCURRENCY} via semaphore.")
    try:
        agent_instance = SelectorAgent(
            html_content=html_content,
            base_url=BASE_URL,
            dom_representation=dom_content,
            semaphore=agent_semaphore,  # Pass semaphore directly
        )
    except ValueError as ve:
        print(f"[bold red]Initialization Error:[/bold red] {ve}")
        return
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to instantiate SelectorAgent: {e}")
        return

    # --- Generate Requests and Start Agent Tasks Concurrently ---
    print("\n🧠 Generating REGION requests and starting agent tasks concurrently...")
    # Call the refactored function, removing dom_content argument
    tasks_with_context: list[
        tuple[asyncio.Task, AgentRequest]
    ] = await create_and_start_agent_tasks(
        proposal=proposal,
        agent_instance=agent_instance,
    )

    if not tasks_with_context:
        print("[bold yellow]Warning:[/bold yellow] No agent tasks were created.")
        return

    # --- Wait for and Gather Agent Task Results ---
    print(f"\n⏳ Waiting for {len(tasks_with_context)} agent tasks to complete...")
    # Gather results from the tasks created by the function
    results_with_exceptions = await asyncio.gather(
        *[task for task, _ in tasks_with_context], return_exceptions=True
    )

    # --- Process Results ---
    print("\n🏁 Processing results...")
    success_count = 0
    failure_count = 0
    results_store: dict[
        tuple[int, Optional[str]], tuple[AgentRequest, AgentResult | Exception]
    ] = {}

    for i, result in enumerate(results_with_exceptions):
        # Get the corresponding original request from the context list
        original_request = tasks_with_context[i][1]
        request_key = (original_request.region_id, original_request.metadata_key)

        print(
            f"\n--- Result for Request (Type: {original_request.request_type}, Region: {original_request.region_id}"
            f"{f', Key: {original_request.metadata_key}' if original_request.metadata_key else ''}) ---"
        )
        print(f"  Description: {original_request.target_description[:100]}...")
        if original_request.proposed_selector_hint:
            print(f"  Proposed Hint: {original_request.proposed_selector_hint}")

        if isinstance(result, Exception):
            failure_count += 1
            print("  [bold red]Status: FAILED (Exception)[/bold red]")
            print(f"  Error: {type(result).__name__}: {result}")
            results_store[request_key] = (original_request, result)
        elif isinstance(result, AgentResult):
            results_store[request_key] = (original_request, result)
            # Evaluate success based on agent finding a unique element without error
            verification_successful = (
                result.final_verification.element_count == 1 and not result.final_verification.error
            )
            # Note: extraction_ok is removed as status depends on context below

            print(f"  Proposed Selector: {result.proposed_selector}")
            print(f"  Reasoning: {result.reasoning[:100]}...")
            print(
                f"  Verification: Count={result.final_verification.element_count}, Error='{result.final_verification.error}', FoundText={result.final_verification.target_text_found_in_any_match}"
            )

            # Print extracted data (including HTML/MD if available)
            if result.extraction_result.error:
                # This error is from the ORIGINAL request if one was made, not the context extraction
                print(f"  Extraction Error (Original Request): {result.extraction_result.error}")
            else:
                if result.attribute_extracted:
                    print(
                        f"  Extracted '{result.attribute_extracted}': {result.extraction_result.extracted_attribute_value}"
                    )
                if result.text_extracted_flag:
                    print(f"  Extracted Text: {result.extraction_result.extracted_text}")

            # Print HTML/MD context if present (should be populated for successful region selection)
            # if result.extraction_result.extracted_html:
            #     print("  Extracted HTML:")
            #     print(
            #         Syntax(
            #             result.extraction_result.extracted_html,
            #             "html",
            #             theme="default",
            #             line_numbers=False,
            #         )
            #     )
            if result.extraction_result.extracted_markdown:
                print("  Extracted Markdown:")
                print(Markdown(result.extraction_result.extracted_markdown))

            # Determine overall status - Refined Logic
            if verification_successful:
                if original_request.request_type == "region_selection":
                    # For regions, success now simply means verification ok, agent handled size internally
                    success_count += 1
                    print("  [bold green]Status: SUCCESS (Region Selected)[/bold green]")
                # Add elif for metadata requests here if/when re-enabled
                # elif original_request.request_type == "metadata_extraction": ...
                else:  # Should not happen with current logic, but handle defensively
                    failure_count += 1
                    print(
                        f"  [bold red]Status: FAILED (Unknown request type: {original_request.request_type})[/bold red]"
                    )
            else:  # verification failed
                failure_count += 1
                print("  [bold red]Status: FAILED (Verification Failed)[/bold red]")

        else:  # Handle case where result is not Exception or AgentResult
            failure_count += 1
            print("  [bold red]Status: FAILED (Unexpected Result Type)[/bold red]")
            print(f"  Result: {type(result)}")
            results_store[request_key] = (
                original_request,
                TypeError(f"Unexpected result type: {type(result)}"),
            )

    # --- Final Summary ---
    print("\n🏁 Final Summary 🏁")
    total_requests = len(tasks_with_context)  # Use the tasks list length
    print(f"Total Requests Attempted: {total_requests}")
    print(f"[green]Successful Requests:[/green] {success_count}")
    print(f"[red]Failed Requests:[/red] {failure_count}")
    # TODO: Add more sophisticated aggregation/output based on results_store

    # Example: Save results to JSON
    try:
        output_results_path = sample_dir / "agent_results.json"
        # Need to handle non-serializable Exceptions if saving raw results
        serializable_results = {}
        for key, (req, res) in results_store.items():
            res_data = None
            if isinstance(res, AgentResult):
                res_data = res.model_dump(mode="json")
            elif isinstance(res, Exception):
                res_data = {"error": f"{type(res).__name__}: {res}"}
            else:
                res_data = {"error": f"Unknown result type: {type(res)}"}

            serializable_results[f"region_{key[0]}_key_{key[1]}"] = {
                "request": req.__dict__,  # Convert AgentRequest dataclass
                "result": res_data,
            }

        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_results_path}")
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

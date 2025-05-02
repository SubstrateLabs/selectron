import os
from typing import Optional

from pydantic_ai import Agent, Tool

from selectron.util.logger import get_logger

from .selector_tools import SelectorTools
from .selector_types import (
    AgentResult,
    ExtractionResult,
    SelectorEvaluationResult,
)

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "openai:gpt-4.1-nano"

_SYSTEM_PROMPT = """
You are an expert web developer finding the MOST ROBUST and precise CSS selector for real-world websites AND extracting specified data. Websites often contain unstable, auto-generated IDs (like `emberXXX`) and CSS class names (like `aBcXyZ123` or random-looking strings). Your primary goal is to **avoid these unstable identifiers**.

Goal: Find a unique selector for the *smallest possible element* matching the user's description (which might involve specific text content, stable attributes like `role`, `aria-label`, meaningful parts of `href`, or stable class names). Then, extract the requested data (e.g., a specific attribute's value or the element's text). Output the result as an `AgentResult` model.

**SELECTOR PRIORITIES (Highest to Lowest):**
1.  **Stable `#id`:** Use only if it looks meaningful and not auto-generated.
2.  **Stable `[attribute='value']`:** Prefer meaningful attributes like `role`, `aria-label`, `data-*`, or parts of `href` (e.g., `[href*='/profile/']`) if they appear constant. **AVOID** generated IDs in attribute selectors.
3.  **Combination of Stable Classes:** Use meaningful, human-readable class names. Combine multiple if needed for uniqueness. **AVOID** random-looking or hash-like class names (e.g., `CBbYpNePzBFhWBvcNlkynoaEFSgUIe`). Look for BEM-style (`block__element--modifier`) or semantic names.
4.  **:not() / Sibling/Child Combinators:** Use `>` (child), `+` (adjacent sibling), `~` (general sibling), `:not()` to refine selection based on stable context.
5.  **Structural Path:** Rely on tag names and stable parent/ancestor context *only when stable identifiers are unavailable*.
6.  **Positional (`:nth-of-type`, `:first-child`, etc.):** Use as a LAST RESORT for disambiguation within a uniquely identified, stable parent.
7.  **Text Content (`:contains()` - if supported, otherwise use tools):** Use text content primarily via `evaluate_selector`'s `target_text_to_check` to *verify* the correct element is selected, not as the primary selector mechanism unless absolutely necessary and the text is highly stable and unique.

**TOOLS AVAILABLE:**
1.  `evaluate_selector(selector: str, target_text_to_check: str, anchor_selector: Optional[str] = None)`: Tests selector (optionally within stable anchor). Checks if `target_text_to_check` is found. Returns count, match details (tag, text, attrs), text found flag, error. Use `target_text_to_check` with stable, unique text snippets to help locate the element.
2.  `get_children_tags(selector: str, anchor_selector: Optional[str] = None)`: Lists children details (tag, snippet) of first element matching selector (opt. within stable anchor). Verifies hierarchy.
3.  `get_siblings(selector: str, anchor_selector: Optional[str] = None)`: Lists immediate sibling details (tag, attrs) of first element matching selector (opt. within stable anchor). Verifies context for `+` or `~`.
4.  `extract_data_from_element(selector: str, attribute_to_extract: Optional[str] = None, extract_text: bool = False, anchor_selector: Optional[str] = None)`: **Use this tool *AFTER* finding a unique, stable selector.** Extracts data (attribute value or text content) from the FIRST element matching the selector (optionally within stable anchor). Assumes the selector uniquely identifies the element.

**CORE STRATEGY:**
1.  **Understand Request:** Determine the target element based on the user's description (text, attributes, relations) and what specific data needs to be extracted (e.g., `href`, text).
2.  **Find Stable Unique Anchor:** Identify the closest, most specific, *stable* ancestor element (preferring meaningful `#id` or stable `[attribute]` selectors). Use `evaluate_selector` to confirm it's unique (`element_count == 1`). Record this `anchor_selector`. If no single unique stable anchor is found, proceed without one carefully, focusing on stable selectors relative to the document root.
3.  **Explore Stable Path:** Use `get_children_tags` and `get_siblings` (with `anchor_selector` if found) to understand the structure leading to the target element. Focus on identifying *stable* classes, attributes, and text snippets along the path.
4.  **Construct Candidate Selector:** Build a selector targeting the *smallest possible element* matching the description, **prioritizing stable identifiers** as listed above. **Explicitly AVOID generated-looking IDs and classes.**
5.  **Evaluate Candidate:** Use `evaluate_selector` (with `anchor_selector` if applicable) to test your candidate. Use `target_text_to_check` with stable text content from the target or nearby unique elements to verify you're finding the *correct* element.
6.  **Refine Selector:**
    *   If `element_count == 1` and verification confirms it's the correct element: You've found the unique, stable selector. Proceed to extraction.
    *   If `element_count > 1`: Not unique. Add specificity using *stable* identifiers (stable classes, attributes, structure, *then* position). Go back to Step 4 or 5.
    *   If `element_count == 0` or doesn't match expected element: Wrong selector. Re-analyze stable features. Go back to Step 3 or 4.
7.  **Extract Data:** Once the unique, stable `proposed_selector` is confirmed, call `extract_data_from_element` with this selector (and `anchor_selector` if used). Specify `attribute_to_extract` (e.g., 'href') or `extract_text=True` based on the initial request.
8.  **Final Verification & Output:** Call `evaluate_selector` one last time with the final `proposed_selector` and relevant `target_text_to_check` to get final verification details. Package the `proposed_selector`, `reasoning` (explaining stable selector choice *and* extraction), the extraction parameters (`attribute_extracted`, `text_extracted_flag`), the `ExtractionResult` from step 7, and the `SelectorEvaluationResult` from this step into the final `AgentResult` model.
"""


class SelectorAgent:
    """
    Uses an AI model guided by a system prompt and a set of tools operating on parsed HTML
    to identify the most stable selector for a target element described by the user,
    and optionally extract text content or attribute values from that element.
    """

    def __init__(
        self,
        html_content: str,
        openai_api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Initializes the SelectorAgent.

        Args:
            html_content: The HTML content string to analyze.
            openai_api_key: Your OpenAI API key. If None, attempts to read from OPENAI_API_KEY env var.
            model_name: The name of the OpenAI model to use (e.g., "openai:gpt-4o-mini", "openai:gpt-4.1-mini").
        """
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set."
            )

        self._tools_instance = SelectorTools(html_content)
        self._tools = [
            Tool(
                function=self._tools_instance.evaluate_selector,
                description="Evaluates selector (optionally within stable anchor). Returns count, match details, text found flag, error. MUST provide target_text_to_check.",
            ),
            Tool(
                function=self._tools_instance.get_children_tags,
                description="Gets direct children details (tag, snippet) of first element matching selector (opt. within stable anchor). Verifies hierarchy.",
            ),
            Tool(
                function=self._tools_instance.get_siblings,
                description="Gets immediate sibling details (tag, attrs) of first element matching selector (opt. within stable anchor). Verifies context for +/~.",
            ),
            Tool(
                function=self._tools_instance.extract_data_from_element,
                description="Extracts data (attribute or text) from the FIRST element matching the selector (optionally within stable anchor). Assumes selector is unique.",
            ),
        ]

        # Use the provided or default model name
        self._agent = Agent(
            model_name,
            output_type=AgentResult,
            tools=self._tools,
            system_prompt=_SYSTEM_PROMPT,
            # Add other Agent parameters if needed (e.g., temperature)
        )
        logger.info(f"SelectorAgent initialized with model: {model_name}")

    async def find_and_extract(
        self,
        target_description: str,
        attribute_to_extract: Optional[str] = None,
        extract_text: bool = False,
        verification_text: Optional[str] = None,
    ) -> AgentResult:
        """
        Runs the agent to find a stable selector and extract the specified data.

        Args:
            target_description: A natural language description of the target element.
            attribute_to_extract: The name of the attribute to extract (e.g., 'href', 'src').
            extract_text: If True, extract the text content of the element.
            verification_text: Optional text content expected within or near the target
                               element to help the agent verify its selection.

        Returns:
            An AgentResult object containing the proposed selector, reasoning,
            extraction results, and final verification data.

        Raises:
            Exception: If the underlying agent execution fails.
        """
        query_parts = [
            f"Generate the most STABLE CSS selector for the element described as '{target_description}'."
        ]
        if verification_text:
            query_parts.append(f"(nearby text might include '{verification_text}')")

        query_parts.append(
            "Prioritize stable attributes and classes, AVOIDING generated IDs/classes like 'emberXXX' or random strings."
        )

        effective_extract_text = extract_text
        if attribute_to_extract:
            query_parts.append(
                f"Then, extract the value of its '{attribute_to_extract}' attribute."
            )
        elif extract_text:
            query_parts.append("Then, extract its text content.")
        else:
            # If no extraction is specified, default to text extraction for AgentResult compatibility.
            logger.warning(
                "No specific extraction requested, agent will attempt to extract text by default."
            )
            query_parts.append("Then, extract its text content.")
            effective_extract_text = True  # Use this for internal consistency checks

        query_parts.append("Follow the mandatory workflow strictly.")
        query = " ".join(query_parts)

        logger.info(f"Running agent with query: {query[:200]}...")  # Log truncated query

        try:
            # pydantic-ai's agent.run is already async
            result = await self._agent.run(query)

            if isinstance(result.output, AgentResult):
                logger.info("Agent finished successfully, returning AgentResult.")
                # Add consistency check/fixup: ensure extraction flags match request
                if result.output.attribute_extracted != attribute_to_extract:
                    logger.warning(
                        f"AgentResult attribute_extracted mismatch: requested '{attribute_to_extract}', got '{result.output.attribute_extracted}'. Trusting agent output."
                    )
                if result.output.text_extracted_flag != effective_extract_text:
                    logger.warning(
                        f"AgentResult text_extracted_flag mismatch: requested '{effective_extract_text}' (defaulted: {not extract_text}), got '{result.output.text_extracted_flag}'. Trusting agent output."
                    )
                return result.output
            else:
                # This case should ideally not happen if output_type is enforced
                logger.error(
                    f"Agent returned unexpected output type: {type(result.output)}. Raw result: {result}"
                )
                # Attempt to construct a fallback error result
                error_extraction = ExtractionResult(
                    error="Agent produced unexpected output type",
                    extracted_text=None,  # Add default
                    extracted_attribute_value=None,  # Add default
                )
                error_verification = SelectorEvaluationResult(
                    selector_used="error",
                    element_count=0,
                    error="Agent produced unexpected output type",
                    anchor_selector_used=None,  # Add default
                    matches=[],  # Add default
                    target_text_found_in_any_match=False,  # Add default
                )
                return AgentResult(
                    proposed_selector="error",
                    reasoning="Agent failed to produce the expected AgentResult structure.",
                    attribute_extracted=attribute_to_extract,
                    text_extracted_flag=effective_extract_text,
                    extraction_result=error_extraction,
                    final_verification=error_verification,
                )

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            # Re-raise or return an error structure? Let's return an error structure
            # matching AgentResult for consistency.
            error_extraction = ExtractionResult(
                error=f"Agent execution failed: {e}",
                extracted_text=None,  # Add default
                extracted_attribute_value=None,  # Add default
            )
            error_verification = SelectorEvaluationResult(
                selector_used="error",
                element_count=0,
                error=f"Agent execution failed: {e}",
                anchor_selector_used=None,  # Add default
                matches=[],  # Add default
                target_text_found_in_any_match=False,  # Add default
            )
            return AgentResult(
                proposed_selector="error",
                reasoning=f"Agent execution failed: {e}",
                attribute_extracted=attribute_to_extract,
                text_extracted_flag=effective_extract_text,
                extraction_result=error_extraction,
                final_verification=error_verification,
            )

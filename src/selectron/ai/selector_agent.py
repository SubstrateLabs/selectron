import asyncio
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

# DEFAULT_MODEL_NAME = "openai:gpt-4.1"
DEFAULT_MODEL_NAME = "anthropic:claude-3-7-sonnet-latest"

_SYSTEM_PROMPT_BASE = """
You are an expert web developer finding the MOST ROBUST and precise CSS selector for real-world websites AND extracting specified data. Websites often contain unstable, auto-generated IDs (like `emberXXX`) and CSS class names (like `aBcXyZ123` or random-looking strings). Your primary goal is to **avoid these unstable identifiers**.

Goal: Find a unique selector for the ***smallest possible and most specific element*** matching the user's description (which might involve specific text content, stable attributes like `role`, `aria-label`, meaningful parts of `href`, or stable class names). When asked for a container element, prioritize the most immediate parent that accurately encompasses the described content, preferring semantic tags like `<article>`, `<section>`, `<aside>`, `<li>` over generic `<div>` or `<span>` unless the generic tags have highly stable and unique attributes.
Then, extract the requested data (e.g., a specific attribute's value or the element's text). Output the result as an `AgentResult` model.

**SELECTOR PRIORITIES (Highest to Lowest):**
1.  **Stable `#id`:** Use only if it looks meaningful and not auto-generated.
2.  **Stable `[attribute='value']`:** Prefer meaningful attributes like `role`, `aria-label`, `data-*`, or parts of `href` (e.g., `[href*='/profile/']`) if they appear constant. **AVOID** generated IDs in attribute selectors.
3.  **Combination of Stable Classes:** Use meaningful, human-readable class names. Combine multiple if needed for uniqueness. **AVOID** random-looking or hash-like class names (e.g., `CBbYpNePzBFhWBvcNlkynoaEFSgUIe`). Look for BEM-style (`block__element--modifier`) or semantic names.
4.  **:not() / Sibling/Child Combinators:** Use `>` (child), `+` (adjacent sibling), `~` (general sibling), `:not()` to refine selection based on stable context.
5.  **Structural Path:** Rely on tag names and stable parent/ancestor context *only when stable identifiers are unavailable*.
6.  **Positional (`:nth-of-type`, `:first-child`, etc.):** Use as a LAST RESORT for disambiguation within a uniquely identified, stable parent.
7.  **Text Content (`:contains()` - if supported, otherwise use tools):** Use text content primarily via `evaluate_selector`'s `target_text_to_check` to *verify* the correct element is selected. **DO NOT use `:contains` or similar text-matching pseudo-classes (like `:has(:contains(...))`) in the final `proposed_selector`** unless absolutely NO other stable identifier (attribute, class, structure) can uniquely identify the element, and the text content itself is GUARANTEED to be stable and unique.

**TOOLS AVAILABLE:**
1.  `evaluate_selector(selector: str, target_text_to_check: str, anchor_selector: Optional[str] = None, max_html_length: Optional[int] = None)`: Tests selector (optionally within stable anchor). Checks if `target_text_to_check` is found. **If `max_html_length` is provided and count is 1, checks element HTML length.** Returns count, match details, text found flag, `error`, and `size_validation_error`. Use `target_text_to_check` with stable, unique text snippets to help locate the element.
2.  `get_children_tags(selector: str, anchor_selector: Optional[str] = None)`: Lists children details (tag, snippet) of first element matching selector (opt. within stable anchor). Verifies hierarchy.
3.  `get_siblings(selector: str, anchor_selector: Optional[str] = None)`: Lists immediate sibling details (tag, attrs) of first element matching selector (opt. within stable anchor). Verifies context for `+` or `~`.
4.  `extract_data_from_element(selector: str, attribute_to_extract: Optional[str] = None, extract_text: bool = False, anchor_selector: Optional[str] = None)`: **Use this tool *AFTER* finding a unique, stable selector.** Extracts data (attribute value or text content) from the FIRST element matching the selector (optionally within stable anchor). Assumes the selector uniquely identifies the element.

**CORE STRATEGY:**
1.  **Understand Request:** Determine the target element based on the user's description (text, attributes, relations) and what specific data needs to be extracted (e.g., `href`, text). **Pay close attention to whether a specific item or its container is requested.**
2.  **Find Stable Unique Anchor:** Identify the closest, most specific, *stable* ancestor element (preferring meaningful `#id` or stable `[attribute]` selectors). Use `evaluate_selector` to confirm it's unique (`element_count == 1`). Record this `anchor_selector`. If no single unique stable anchor is found, proceed without one carefully, focusing on stable selectors relative to the document root.
3.  **Explore Stable Path:** Use `get_children_tags` and `get_siblings` (with `anchor_selector` if found) to understand the structure leading to the target element. Focus on identifying *stable* classes, attributes, and text snippets along the path.
4.  **Construct Candidate Selector:** Build a selector targeting the ***smallest possible and most specific element*** matching the description, prioritizing stable identifiers as listed above. **Avoid overly broad selectors like generic `div:has(...)` if a more specific container like `<article>` or a direct parent with stable attributes exists.** Explicitly AVOID generated-looking IDs and classes.
5.  **Evaluate Candidate:** Use `evaluate_selector` (with `anchor_selector` if applicable) to test your candidate. **ALWAYS provide a `max_html_length` (e.g., 5000)** to check size. Use `target_text_to_check` with stable text content from the target or nearby unique elements to verify you're finding the *correct* element.
6.  **Refine Selector:**
    *   Check the result from `evaluate_selector`:
        *   If `error` is present OR `size_validation_error` is present: The evaluation failed or the element is too large/selector too broad. Go back to Step 3 or 4 to find a better, more specific selector.
        *   If `element_count == 1` and `size_validation_error` is `None` and verification confirms it's the correct element: You've found the unique, stable, and appropriately sized selector. Proceed to extraction (Step 7).
        *   If `element_count > 1`: Not unique. Add specificity using *stable* identifiers (stable classes, attributes, structure, *then* position). Go back to Step 4 or 5.
        *   If `element_count == 0`: Wrong selector. Re-analyze stable features. Go back to Step 3 or 4.
7.  **Extract Data:** Once the unique, stable, validated `proposed_selector` is confirmed, call `extract_data_from_element` with this selector (and `anchor_selector` if used). Specify `attribute_to_extract` (e.g., 'href') or `extract_text=True` based on the initial request.
8.  **Final Verification & Output:** Call `evaluate_selector` one last time with the final `proposed_selector` and relevant `target_text_to_check` to get final verification details. Package the `proposed_selector`, `reasoning` (explaining stable selector choice *and* extraction), the extraction parameters (`attribute_extracted`, `text_extracted_flag`), the `ExtractionResult` from step 7, and the `SelectorEvaluationResult` from this step into the final `AgentResult` model.
"""

_DOM_CONTEXT_PROMPT_SECTION = """

**ADDITIONAL CONTEXT: SIMPLIFIED DOM REPRESENTATION**
A simplified text representation of the DOM structure (derived from `dom.txt`) is provided below. It uses a format like `[node_id]<tag attributes...> text_snippet`. You can use this simplified view to help understand the stable structure and relationships between elements when choosing identifiers. However, remember that your final `proposed_selector` **MUST** work on the full HTML and be verified using the provided tools (`evaluate_selector`, etc.). Do not attempt to directly query this text representation with tools.

--- SIMPLIFIED DOM START ---
{dom_representation}
--- SIMPLIFIED DOM END ---
"""


class SelectorAgent:
    def __init__(
        self,
        html_content: str,
        base_url: str,
        model_name: str = DEFAULT_MODEL_NAME,
        dom_representation: Optional[str] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ):
        """
        Args:
            html_content: The HTML content string to analyze.
            base_url: The base URL of the page, used for resolving relative URLs.
            model_name: The Pydantic AI model name (e.g., "openai:gpt-4o-mini", "openai:gpt-4.1-mini").
            dom_representation: Optional simplified text representation of the DOM.
            semaphore: Optional asyncio.Semaphore to limit concurrent agent runs.
        """
        if not base_url:
            raise ValueError("base_url must be provided.")

        self._tools_instance = SelectorTools(html_content, base_url)
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

        # Construct the system prompt, adding DOM context if provided
        system_prompt = _SYSTEM_PROMPT_BASE
        if dom_representation:
            # Limit DOM representation size to avoid excessive prompt length
            # TODO: Make this limit configurable or smarter?
            max_dom_len = 10000
            truncated_dom = dom_representation[:max_dom_len]
            if len(dom_representation) > max_dom_len:
                truncated_dom += "\n... (truncated)"
                logger.warning(f"DOM representation truncated to {max_dom_len} characters.")

            system_prompt += _DOM_CONTEXT_PROMPT_SECTION.format(dom_representation=truncated_dom)
            logger.info("Added simplified DOM representation to system prompt.")
        else:
            logger.info("No simplified DOM representation provided.")

        # Use the provided or default model name
        self._agent = Agent(
            model_name,
            output_type=AgentResult,
            tools=self._tools,
            system_prompt=system_prompt,
            # Add other Agent parameters if needed (e.g., temperature)
        )
        logger.info(f"SelectorAgent initialized with model: {model_name}")
        self._semaphore = semaphore

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

        effective_extract_text = extract_text  # Start with the user's explicit request
        if attribute_to_extract:
            query_parts.append(
                f"Then, extract the value of its '{attribute_to_extract}' attribute."
            )
        elif extract_text:  # Only if explicitly requested
            query_parts.append("Then, extract its text content.")
        else:
            # If no extraction is specified, DO NOT ask the agent to extract anything.
            # The agent's output model (AgentResult) includes flags for what was extracted,
            # so we don't need to force an extraction action here.
            logger.info(
                "No specific extraction requested (attribute or text). Agent will focus on selection."
            )
            # Explicitly tell the agent NOT to extract.
            query_parts.append(
                "You MUST NOT call the `extract_data_from_element` tool for this request."
            )
            # effective_extract_text remains False (from initialization)

        query_parts.append("Follow the mandatory workflow strictly.")
        query = " ".join(query_parts)

        logger.info(f"Running agent with query: {query[:200]}...")  # Log truncated query

        # Acquire semaphore if it exists before running the agent logic
        if self._semaphore:
            await self._semaphore.acquire()
            logger.debug("Semaphore acquired for agent run.")

        try:
            # Pass usage_limits to the internal agent run
            agent_run_result = await self._agent.run(query)

            if isinstance(agent_run_result.output, AgentResult):
                # Get the initial result from the agent
                agent_result: AgentResult = agent_run_result.output
                logger.info("Agent finished successfully, returning AgentResult.")

                # --- Always attempt HTML/MD extraction on successful selection --- #
                if (
                    agent_result.proposed_selector != "error"
                    and agent_result.final_verification.element_count == 1
                ):
                    logger.info(
                        f"Successful selection ({agent_result.proposed_selector}), attempting HTML/MD context extraction."
                    )
                    try:
                        # Call extraction tool ONLY for HTML/MD context
                        context_extraction: ExtractionResult = await self._tools_instance.extract_data_from_element(
                            selector=agent_result.proposed_selector,
                            attribute_to_extract=None,  # Don't request attribute again
                            extract_text=False,  # Don't request text again
                            anchor_selector=agent_result.final_verification.anchor_selector_used,
                        )
                        # Update the original result object by creating a new one
                        # Preserve original text/attribute data and error status
                        updated_extraction_res = ExtractionResult(
                            extracted_text=agent_result.extraction_result.extracted_text,
                            extracted_attribute_value=agent_result.extraction_result.extracted_attribute_value,
                            extracted_markdown=context_extraction.extracted_markdown,
                            extracted_html=context_extraction.extracted_html,
                            error=agent_result.extraction_result.error,  # Preserve original error
                        )
                        agent_result.extraction_result = updated_extraction_res  # Assign new object

                        if context_extraction.error:
                            logger.warning(
                                f"Context extraction encountered an error: {context_extraction.error}"
                            )
                            # Optionally append to existing error or log? For now, just log.

                    except Exception as context_err:
                        logger.error(
                            f"Error during internal context extraction: {context_err}",
                            exc_info=True,
                        )
                # --- End HTML/MD extraction --- #

                # Add consistency check/fixup for originally requested extraction
                if agent_result.attribute_extracted != attribute_to_extract:
                    logger.warning(
                        f"AgentResult attribute_extracted mismatch: requested '{attribute_to_extract}', got '{agent_result.attribute_extracted}'. Trusting agent output."
                    )
                if agent_result.text_extracted_flag != effective_extract_text:
                    logger.warning(
                        f"AgentResult text_extracted_flag mismatch: requested '{effective_extract_text}', got '{agent_result.text_extracted_flag}'. Trusting agent output."
                    )

                # Return the potentially updated agent_result
                return agent_result
            else:
                # This case should ideally not happen if output_type is enforced
                logger.error(
                    f"Agent returned unexpected output type: {type(agent_run_result.output)}. Raw result: {agent_run_result}"
                )
                # Attempt to construct a fallback error result
                error_extraction = ExtractionResult(
                    error="Agent produced unexpected output type",
                    extracted_text=None,
                    extracted_attribute_value=None,
                    extracted_markdown=None,
                    extracted_html=None,
                )
                error_verification = SelectorEvaluationResult(
                    selector_used="error",
                    element_count=0,
                    error="Agent produced unexpected output type",
                    anchor_selector_used=None,  # Add default
                    matches=[],  # Add default
                    target_text_found_in_any_match=False,  # Add default
                    size_validation_error=None,  # Add default
                )
                return AgentResult(
                    proposed_selector="error",
                    reasoning="Agent failed to produce the expected AgentResult structure.",
                    attribute_extracted=attribute_to_extract,
                    text_extracted_flag=effective_extract_text,  # Use the calculated flag
                    extraction_result=error_extraction,
                    final_verification=error_verification,
                )

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            # Construct error result matching AgentResult
            error_extraction = ExtractionResult(
                error=f"Agent execution failed: {e}",
                extracted_text=None,
                extracted_attribute_value=None,
                extracted_markdown=None,
                extracted_html=None,
            )
            # Need SelectorEvaluationResult here for final_verification
            error_verification = SelectorEvaluationResult(
                selector_used="error",
                element_count=0,
                error=f"Agent execution failed: {e}",
                anchor_selector_used=None,
                matches=[],
                target_text_found_in_any_match=False,
                size_validation_error=None,  # Add default
            )
            return AgentResult(
                proposed_selector="error",
                reasoning=f"Agent execution failed: {e}",
                attribute_extracted=attribute_to_extract,
                text_extracted_flag=effective_extract_text,
                extraction_result=error_extraction,
                final_verification=error_verification,
            )
        finally:
            # Release semaphore if it exists
            if self._semaphore:
                self._semaphore.release()
                logger.debug("Semaphore released for agent run.")

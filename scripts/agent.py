import asyncio
import os
from typing import Literal, Optional

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool

from selectron.util.logger import get_logger

logger = get_logger(__name__)

# More complex HTML
DUMMY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Complex Agent Test Page</title>
    <style>
        .item { border: 1px solid #eee; padding: 5px; margin: 5px; }
        .highlight { background-color: yellow; }
        .data-point { font-style: italic; }
        .section-a .item { color: blue; }
        .section-b .item { color: green; }
    </style>
</head>
<body>
    <header>
        <h1>Complex Page</h1>
        <nav>Skip to: <a href="#section-a">Section A</a> | <a href="#section-b">Section B</a></nav>
    </header>

    <div class="container" role="main">
        <section id="section-a" class="section-a">
            <h2>Section A</h2>
            <p>Some introductory text for section A.</p>
            <div class="item-list">
                <div class="item">Item A1 <span class="data-point">Data A1</span></div>
                <div class="item highlight">Item A2 (Highlighted) <span class="data-point">Data A2</span></div>
                <div class="item">Item A3 <span class="data-point">Data A3</span></div>
                <div class="sub-section">
                    <h3>Nested List A</h3>
                    <ul>
                        <li>Sub Item A.1</li>
                        <li class="target-sibling">Sub Item A.2 (Sibling of Target)</li>
                        <li>Sub Item A.3 <span class="data-point">Data A.3.1</span></li>
                    </ul>
                </div>
            </div>
        </section>

        <section id="section-b" class="section-b">
            <h2>Section B</h2>
            <p>Section B has similar items.</p>
            <div class="item-list">
                <div class="item">Item B1 <span class="data-point">Data B1</span></div>
                <div class="item">Item B2 <span class="data-point">Data B2</span></div>
                <div class="sub-section">
                    <h3>Nested List B</h3>
                    <ul>
                        <li>Sub Item B.1</li>
                        <li>Sub Item B.2</li>
                        <li class="data-point item">Sub Item B.3 (Target)</li>
                    </ul>
                </div>
                 <div class="item highlight">Item B3 (Highlighted) <span class="data-point">Data B3</span></div>
            </div>
        </section>
    </div>

    <footer>
        <p>End of complex page. &copy; 2024</p>
    </footer>
</body>
</html>
"""

# --- Pydantic Models ---


# NEW: Model for detailed match info
class MatchDetail(BaseModel):
    tag_name: str = Field(..., description="Tag name of the matched element.")
    text_content: Optional[str] = Field(None, description="Truncated text content (max 150 chars).")
    attributes: dict[str, str] = Field({}, description="Dictionary of the element's attributes.")
    # html_snippet: str = Field(..., description="Truncated HTML snippet (max 250 chars).") # Optional: Could add this too


# UPDATED: Result model for evaluate_selector
class SelectorEvaluationResult(BaseModel):
    """Result of evaluating a CSS selector, potentially within an anchor context."""

    selector_used: str = Field(...)
    anchor_selector_used: Optional[str] = Field(
        None, description="The anchor selector used, if any."
    )
    element_count: int = Field(
        ..., description="The total number of elements found by the selector."
    )
    matches: list[MatchDetail] = Field(
        [], description="Details of the first few matched elements (up to 5)."
    )
    target_text_found_in_any_match: bool = Field(
        False, description="True if TARGET_TEXT was found in any matched element's text_content."
    )
    error: Optional[str] = Field(None)


class ChildDetail(BaseModel):
    tag_name: str = Field(..., description="The tag name of the child element.")
    html_snippet: str = Field(
        ..., description="A truncated HTML snippet of the child element (max 100 chars)."
    )


class ChildrenTagsResult(BaseModel):
    """Result of getting direct children details, potentially within an anchor context."""

    selector_used: str = Field(...)  # Selector for the parent element relative to anchor/doc
    anchor_selector_used: Optional[str] = Field(
        None, description="The anchor selector used, if any."
    )
    parent_found: bool = Field(..., description="Whether the selector found a parent element.")
    children_details: Optional[list[ChildDetail]] = Field(
        None, description="List of details (tag, snippet) of direct children, if parent found."
    )
    error: Optional[str] = Field(
        None, description="Any error encountered during selection or processing."
    )


class SiblingDetail(BaseModel):
    tag_name: str = Field(...)
    direction: Literal["previous", "next"] = Field(...)
    attributes: dict[str, str] = Field({}, description="Dictionary of the sibling's attributes.")


class SiblingsResult(BaseModel):
    """Result of getting sibling details, potentially within an anchor context."""

    selector_used: str = Field(...)
    anchor_selector_used: Optional[str] = Field(
        None, description="The anchor selector used, if any."
    )
    element_found: bool = Field(
        ..., description="Whether the selector found the reference element."
    )
    siblings: list[SiblingDetail] = Field(
        [], description="Details of immediate preceding and succeeding siblings."
    )
    error: Optional[str] = Field(None, description="Any error encountered.")


class ProposedSelector(BaseModel):
    """The final proposed CSS selector."""

    selector: str = Field(..., description="The CSS selector proposed to find the target element.")
    reasoning: str = Field(..., description="Brief explanation for choosing this selector.")


# --- Tool Definition ---
# Note: Linter noise ignored.
class SelectorTools:
    def __init__(self, html_content: str):
        # Re-add __init__ as tools are stateless now
        self.soup = BeautifulSoup(html_content, "html.parser")
        logger.info("--- HTML Structure Initialized (View full structure if needed) ---")

    async def evaluate_selector(
        self, selector: str, target_text_to_check: str, anchor_selector: Optional[str] = None
    ) -> SelectorEvaluationResult:
        """Evaluates selector (optionally within anchor). Checks if target_text_to_check is found. Returns count, details, text found flag, errors."""
        log_prefix = (
            f"Evaluate Selector ('{selector}'"
            + (f" within '{anchor_selector}'" if anchor_selector else "")
            + ")"
        )
        logger.info(f"{log_prefix}: Starting evaluation for text '{target_text_to_check}'.")

        base_element: BeautifulSoup | Tag | None = self.soup
        if anchor_selector:
            possible_anchors = self.soup.select(anchor_selector)
            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return SelectorEvaluationResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_count=0,
                    matches=[],
                    target_text_found_in_any_match=False,
                    error=error_msg,
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)} elements)."
                logger.warning(f"{log_prefix}: {error_msg}")
                return SelectorEvaluationResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_count=0,
                    matches=[],
                    target_text_found_in_any_match=False,
                    error=error_msg,
                )
            base_element = possible_anchors[0]
            logger.debug(f"{log_prefix}: Anchor found successfully.")

        assert base_element is not None, "Base element for selection cannot be None"

        max_matches_to_detail = 5
        max_text_len = 150
        text_found_flag = False
        try:
            elements = base_element.select(selector)
            count = len(elements)
            match_details: list[MatchDetail] = []

            for i, el in enumerate(elements):
                text = el.get_text(strip=True)
                if target_text_to_check in text:
                    text_found_flag = True
                if len(text) > max_text_len:
                    text = text[:max_text_len] + "..."
                if i < max_matches_to_detail:
                    attrs = {
                        k: " ".join(v) if isinstance(v, list) else v for k, v in el.attrs.items()
                    }
                    match_details.append(
                        MatchDetail(tag_name=el.name, text_content=text, attributes=attrs)
                    )

            result = SelectorEvaluationResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_count=count,
                matches=match_details,
                target_text_found_in_any_match=text_found_flag,
                error=None,
            )
            # Log results at INFO level
            logger.info(
                f"{log_prefix}: Result: Count={result.element_count}, TextFound={result.target_text_found_in_any_match}, MatchesDetailed={len(result.matches)}"
            )
            if result.matches:
                logger.info(
                    f"{log_prefix}: First Match Details: tag='{result.matches[0].tag_name}', text='{result.matches[0].text_content}', attrs={result.matches[0].attributes}"
                )
            return result
        except Exception as e:
            error_msg = f"Evaluation Error: {type(e).__name__}: {e}"
            logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
            return SelectorEvaluationResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_count=0,
                matches=[],
                target_text_found_in_any_match=False,
                error=error_msg,
            )

    async def get_children_tags(
        self, selector: str, anchor_selector: Optional[str] = None
    ) -> ChildrenTagsResult:
        """Gets details (tag name, snippet) of direct children of the FIRST element matched by selector (optionally within anchor)."""
        log_prefix = (
            f"Get Children ('{selector}'"
            + (f" within '{anchor_selector}'" if anchor_selector else "")
            + ")"
        )
        logger.info(f"{log_prefix}: Starting request.")

        base_element: BeautifulSoup | Tag | None = self.soup
        if anchor_selector:
            possible_anchors = self.soup.select(anchor_selector)
            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    children_details=None,
                    error=error_msg,
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)} elements)."
                logger.warning(f"{log_prefix}: {error_msg}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    children_details=None,
                    error=error_msg,
                )
            base_element = possible_anchors[0]
            logger.debug(f"{log_prefix}: Anchor found successfully.")

        assert base_element is not None, "Base element for get_children_tags cannot be None"

        max_snippet_len = 5000
        try:
            parent_element = base_element.select_one(selector)
            if parent_element:
                details_list: list[ChildDetail] = []
                for child in parent_element.find_all(recursive=False):
                    if child.name:  # type: ignore
                        snippet = str(child)
                        if len(snippet) > max_snippet_len:
                            snippet = (
                                snippet[: max_snippet_len // 2]
                                + "..."
                                + snippet[-(max_snippet_len // 2) :]
                            )
                        details_list.append(ChildDetail(tag_name=child.name, html_snippet=snippet))  # type: ignore

                # Log summary of children found
                child_tags_summary = (
                    ", ".join([d.tag_name for d in details_list]) if details_list else "None"
                )
                logger.info(
                    f"{log_prefix}: Result: ParentFound={True}, ChildrenTags=[{child_tags_summary}]"
                )
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=True,
                    children_details=details_list,
                    error=None,
                )
            else:
                # Log results at INFO level
                logger.info(f"{log_prefix}: Result: ParentFound={False}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    children_details=None,
                    error="Parent selector did not match any element within the specified context.",
                )
        except Exception as e:
            error_msg = f"Error getting children details: {type(e).__name__}: {e}"
            logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
            # Log results at INFO level (optional, could stay debug)
            logger.info(f"{log_prefix}: Result: ParentFound={False}, Error occurred.")
            return ChildrenTagsResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                parent_found=False,
                children_details=None,
                error=error_msg,
            )

    async def get_siblings(
        self, selector: str, anchor_selector: Optional[str] = None
    ) -> SiblingsResult:
        """Gets details (tag name, attributes) of immediate siblings of the FIRST element matched (optionally within anchor)."""
        log_prefix = (
            f"Get Siblings ('{selector}'"
            + (f" within '{anchor_selector}'" if anchor_selector else "")
            + ")"
        )
        logger.info(f"{log_prefix}: Starting request.")

        base_element: BeautifulSoup | Tag | None = self.soup
        if anchor_selector:
            possible_anchors = self.soup.select(anchor_selector)
            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    siblings=[],
                    error=error_msg,
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)} elements)."
                logger.warning(f"{log_prefix}: {error_msg}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    siblings=[],
                    error=error_msg,
                )
            base_element = possible_anchors[0]
            logger.debug(f"{log_prefix}: Anchor found successfully.")

        assert base_element is not None, "Base element for sibling search cannot be None"

        try:
            element = base_element.select_one(selector)
            if not element:
                # Log results at INFO level
                logger.info(f"{log_prefix}: Result: ElementFound={False}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    siblings=[],
                    error="Selector did not match any element within the specified context.",
                )

            siblings_details: list[SiblingDetail] = []
            logger.debug(f"{log_prefix}: Reference element found: <{element.name}>")  # type: ignore[attr-defined]

            siblings_summary_list = []  # For logging
            # Previous sibling
            prev_sib = element.find_previous_sibling()
            if prev_sib and prev_sib.name:  # type: ignore[attr-defined]
                attrs = {
                    k: " ".join(v) if isinstance(v, list) else v
                    for k, v in prev_sib.attrs.items()  # type: ignore[attr-defined]
                }  # type: ignore[attr-defined]
                siblings_details.append(
                    SiblingDetail(tag_name=prev_sib.name, direction="previous", attributes=attrs)  # type: ignore[attr-defined]
                )
                logger.debug(
                    f"{log_prefix}: Found Previous Sibling: <{prev_sib.name}> attrs={attrs}"  # type: ignore[attr-defined]
                )
                siblings_summary_list.append(f"prev=<{prev_sib.name}>")  # type: ignore[attr-defined]

            # Next sibling
            next_sib = element.find_next_sibling()
            if next_sib and next_sib.name:  # type: ignore[attr-defined]
                attrs = {
                    k: " ".join(v) if isinstance(v, list) else v
                    for k, v in next_sib.attrs.items()  # type: ignore[attr-defined]
                }
                siblings_details.append(
                    SiblingDetail(tag_name=next_sib.name, direction="next", attributes=attrs)  # type: ignore[attr-defined]
                )
                logger.debug(f"{log_prefix}: Found Next Sibling: <{next_sib.name}> attrs={attrs}")  # type: ignore[attr-defined]
                siblings_summary_list.append(f"next=<{next_sib.name}>")  # type: ignore[attr-defined]

            siblings_summary = ", ".join(siblings_summary_list) if siblings_summary_list else "None"
            # Log summary of siblings found
            logger.info(f"{log_prefix}: Result: ElementFound={True}, Siblings=[{siblings_summary}]")
            return SiblingsResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_found=True,
                siblings=siblings_details,
                error=None,
            )
        except Exception as e:
            error_msg = f"Error getting siblings: {type(e).__name__}: {e}"
            logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
            # Log results at INFO level (optional, could stay debug)
            logger.info(f"{log_prefix}: Result: ElementFound={False}, Error occurred.")
            return SiblingsResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_found=False,
                siblings=[],
                error=error_msg,
            )


# System prompt v16 - Principle-Based Guidance
system_prompt = """
You are an expert web developer finding the MOST ROBUST and precise CSS selector for real-world websites.
Goal: Find a unique selector for the *smallest possible element* containing the **target text provided in the user query**.

**SELECTOR PRIORITIES (Highest to Lowest):** ID > Unique Class Combo > Stable Attributes > :not/Sibling Combinators (if needed) > Structural Path > Positional (:nth) > Text Content (:contains - LAST RESORT).

**TOOLS AVAILABLE:**
1.  `evaluate_selector(selector: str, target_text_to_check: str, anchor_selector: Optional[str] = None)`: Tests selector (optionally within anchor). Checks if `target_text_to_check` is found. Returns count, match details (tag, text, attrs), text found flag, error.
2.  `get_children_tags(selector: str, anchor_selector: Optional[str] = None)`: Lists children details (tag, snippet) of first element matching selector (opt. within anchor). Verifies hierarchy.
3.  `get_siblings(selector: str, anchor_selector: Optional[str] = None)`: Lists immediate sibling details (tag, attrs) of first element matching selector (opt. within anchor). Verifies context for `+` or `~`.

**CORE STRATEGY:**
1.  **Find Unique Anchor:** Identify the closest, most specific, stable ancestor element (preferring `#id`). Use `evaluate_selector(selector=ANCHOR_SELECTOR, target_text_to_check=...)` to confirm it's unique (`element_count == 1`). Record this `anchor_selector`. If no single unique anchor is easily found, proceed without one carefully.
2.  **Explore Path (if using anchor):** Use `get_children_tags` and `get_siblings`, **always providing the verified `anchor_selector`**, to understand the structure between the anchor and the target element. Note useful classes/attributes.
3.  **Construct Candidate Selector:** Build a selector starting from the anchor (if found) down to the target element, using stable identifiers (classes, attributes) found during exploration. Refine to target the *smallest* element containing the text. Add position (`:nth-of-type`) only if necessary for uniqueness within the parent. **Strongly prefer structural selectors over text-based ones like `:contains`.**
4.  **Evaluate Candidate:** Use `evaluate_selector` (with the `anchor_selector` if applicable, and the `target_text_to_check`) to test your candidate.
5.  **Refine or Finalize:**
    *   If `element_count == 1` and `target_text_found_in_any_match == True`: You've likely found the correct selector. Finalize.
    *   If `element_count > 1`: The selector is not unique. Add more specificity (better path, classes, attributes, *then* position) based on the `matches` returned by `evaluate_selector`. Go back to Step 3 or 4.
    *   If `element_count == 0` or `target_text_found_in_any_match == False`: The selector is wrong. Re-analyze the HTML structure and your path assumptions. Go back to Step 2 or 3.
    *   If you get an error (e.g., syntax, repeated selector): Correct the selector and retry evaluation.

**Output the `ProposedSelector` with reasoning explaining the chosen anchor (if any) and path.**
"""

# Instantiate tools globally (stateless now)
selector_tool_instance = SelectorTools(DUMMY_HTML)
tools = [
    Tool(
        function=selector_tool_instance.evaluate_selector,
        description="Evaluates selector (optionally within anchor). Returns count, match details, text found flag, error. MUST provide target_text_to_check.",
    ),
    Tool(
        function=selector_tool_instance.get_children_tags,
        description="Gets direct children details (tag, snippet) of first element matching selector (optionally within anchor). Verifies hierarchy.",
    ),
    Tool(
        function=selector_tool_instance.get_siblings,
        description="Gets immediate sibling details (tag, attrs) of first element matching selector (optionally within anchor). Verifies context for +/~.",
    ),
]

# Create Agent instance (no deps_type needed)
agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=ProposedSelector,
    tools=tools,
    system_prompt=system_prompt,
)


async def main():
    # Define target text locally
    target_text = "Sub Item B.2"

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[bold red]Error:[/bold red] OPENAI_API_KEY environment variable not set.")
        return

    print(f"\n Agent Goal: Find a selector for the element containing text: '{target_text}'")
    print("\n🚀 Running agent...")

    try:
        # No deps needed for run
        result = await agent.run(
            f"Generate a CSS selector for the smallest element containing the text '{target_text}'. Follow the mandatory workflow strictly."
        )

        print("\n✅ Agent finished.")
        print("\n--- Final Proposed Selector ---")
        final_output = result.output
        if final_output and isinstance(final_output, ProposedSelector):
            proposed_selector = final_output.selector
            reasoning = final_output.reasoning
            print(f"Selector: {proposed_selector}")
            print(f"Reasoning: {reasoning}")

            print("\n--- Verifying Agent's Proposed Selector ---")
            # Verification uses the global stateless tool instance
            eval_result = await selector_tool_instance.evaluate_selector(
                proposed_selector, target_text_to_check=target_text
            )
            print("Verification Result:")
            print(f"  Selector Used: {eval_result.selector_used}")
            print(f"  Element Count: {eval_result.element_count}")
            first_match = eval_result.matches[0] if eval_result.matches else None
            if first_match:
                print(f"  First Match Tag: '{first_match.tag_name}'")
                print(f"  First Match Text: '{first_match.text_content}'")
                print(f"  First Match Attrs: {first_match.attributes}")
            else:
                print("  First Match: None")
            print(f"  Target Text Found Flag: {eval_result.target_text_found_in_any_match}")
            if eval_result.error:
                print(f"  Error: {eval_result.error}")

            if eval_result.element_count == 1 and eval_result.target_text_found_in_any_match:
                print(
                    "[bold green]Verification SUCCESSFUL! Text found in unique element.[/bold green]"
                )
            elif eval_result.element_count > 0 and eval_result.target_text_found_in_any_match:
                print(
                    "[bold yellow]Verification PARTIAL: Text found, but selector is NOT unique.[/bold yellow]"
                )
            else:
                print(
                    "[bold red]Verification FAILED! Selector wrong, not unique, or doesn't contain text.[/bold red]"
                )

        else:
            print(
                "[bold yellow]Warning:[/bold yellow] Agent did not produce the expected ProposedSelector output."
            )
            print("\n--- Raw Agent Result ---")
            print(result)

    except Exception as e:
        print(f"\n[bold red]Error during agent execution:[/bold red] {e}")
        # Consider more specific error handling based on pydantic-ai exceptions


if __name__ == "__main__":
    asyncio.run(main())

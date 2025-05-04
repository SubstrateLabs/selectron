import copy
import traceback
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

from selectron.ai.types import (
    ChildDetail,
    ChildrenTagsResult,
    ExtractionResult,
    MatchDetail,
    SelectorEvaluationResult,
    SiblingDetail,
    SiblingsResult,
)
from selectron.util.logger import get_logger

logger = get_logger(__name__)

# Constants for tool verbosity control
DEFAULT_MAX_MATCHES_TO_DETAIL = 5  # Limit number of detailed matches by default
DEFAULT_MAX_CHILDREN_TO_DETAIL = 10  # Limit number of detailed children by default
DEFAULT_MAX_SNIPPET_LENGTH = 300  # Max length for HTML/Markdown snippets
DEFAULT_MAX_HTML_LENGTH_VALIDATION = 5000  # Max length for single element HTML validation


class SelectorTools:
    """Internal class holding the BeautifulSoup instance and tool methods."""

    def __init__(self, html_content: str, base_url: str):
        self.soup = BeautifulSoup(html_content, "html.parser")
        self.base_url = base_url

    def _convert_html_to_markdown(self, element: Tag) -> str:
        """Converts a BeautifulSoup Tag element to a Markdown string."""
        # Basic implementation using markdownify, assumes installed
        try:
            import markdownify

            # Convert the specific element, not just its inner content
            # Use default options, maybe configure later if needed (e.g., heading style)
            md = markdownify.markdownify(str(element), heading_style="ATX")
            # --- Truncate Markdown ---
            if len(md) > DEFAULT_MAX_SNIPPET_LENGTH:
                md = md[:DEFAULT_MAX_SNIPPET_LENGTH] + "..."
            return md.strip()
        except ImportError:
            logger.warning(
                "markdownify library not found. Falling back to plain text for markdown."
            )
            return element.get_text(strip=True)
        except Exception as e:
            logger.error(f"Error during markdown conversion: {e}")
            return f"Error converting to markdown: {e}"

    async def evaluate_selector(
        self,
        selector: str,
        target_text_to_check: str,
        anchor_selector: Optional[str] = None,
        max_html_length: Optional[int] = None,
        max_matches_to_detail: Optional[int] = DEFAULT_MAX_MATCHES_TO_DETAIL,
        return_matched_html: bool = False,
    ) -> SelectorEvaluationResult:
        """Evaluates a CSS selector, checks for text, validates size, and provides rich detail on multiple matches.

        Args:
            selector: The CSS selector to evaluate.
            target_text_to_check: Text content to check for within the matched elements.
            anchor_selector: Optional selector to narrow the search scope.
            max_html_length: Optional max length for single element HTML validation.
            max_matches_to_detail: Max number of matches to include detailed info for. Defaults to {DEFAULT_MAX_MATCHES_TO_DETAIL}.
            return_matched_html: Whether to return the outer HTML of matched elements.

        Returns:
            SelectorEvaluationResult with details.
        """
        log_prefix = (
            f"Evaluate Selector ('{selector}'"
            + (f" for text '{target_text_to_check[:20]}...'" if target_text_to_check else "")
            + (f" within '{anchor_selector}'" if anchor_selector else "")
            + ")"
        )
        base_element: BeautifulSoup | Tag | None = self.soup
        size_validation_error_msg: Optional[str] = None
        text_found_flag = False
        matched_html: list[str] = []

        # --- Handle Anchor Selector --- #
        if anchor_selector:
            try:
                possible_anchors = self.soup.select(anchor_selector)
            except Exception as e:
                error_msg = f"Anchor Selector Syntax Error: {type(e).__name__}: {e}"
                logger.warning(f"{log_prefix}: {error_msg}")
                return SelectorEvaluationResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_count=0,
                    error=error_msg,
                    matches=[],
                    target_text_found_in_any_match=False,
                    size_validation_error=None,
                    feedback_message=None,
                    matched_html_snippets=None,
                )

            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return SelectorEvaluationResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_count=0,
                    error=error_msg,
                    matches=[],
                    target_text_found_in_any_match=False,
                    size_validation_error=None,
                    feedback_message=None,
                    matched_html_snippets=None,
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)})."
                logger.warning(f"{log_prefix}: {error_msg}")
                return SelectorEvaluationResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_count=0,
                    error=error_msg,
                    matches=[],
                    target_text_found_in_any_match=False,
                    size_validation_error=None,
                    feedback_message=None,
                    matched_html_snippets=None,
                )
            base_element = possible_anchors[0]
            logger.debug(f"{log_prefix}: Anchor found successfully.")

        assert base_element is not None, "Base element for evaluate_selector cannot be None"

        try:
            elements = base_element.select(selector)
            count = len(elements)
            match_details: list[MatchDetail] = []

            # --- Populate Match Details (Up to max_matches_to_detail or all if None) --- #
            for i, el in enumerate(elements):
                if not isinstance(el, Tag):
                    continue  # Skip non-Tag elements

                # Capture HTML if requested
                if return_matched_html:
                    try:
                        html_str = str(el)
                        # --- Truncate HTML Snippet ---
                        if len(html_str) > DEFAULT_MAX_SNIPPET_LENGTH:
                            html_str = html_str[:DEFAULT_MAX_SNIPPET_LENGTH] + "..."
                        matched_html.append(html_str)
                    except Exception as html_err:
                        logger.warning(f"Error getting HTML for element {i}: {html_err}")
                        matched_html.append(f"<!-- Error getting HTML: {html_err} -->")

                # Check for target text within the element's raw text
                if target_text_to_check and target_text_to_check in el.get_text(strip=True):
                    text_found_flag = True

                # Get details only if no limit or within limit
                if max_matches_to_detail is None or i < max_matches_to_detail:
                    attrs = {
                        k: " ".join(v) if isinstance(v, list) else v for k, v in el.attrs.items()
                    }
                    # Extract full markdown content using the helper (will be truncated by helper)
                    markdown_content = self._convert_html_to_markdown(el)
                    match_details.append(
                        MatchDetail(
                            tag_name=el.name,
                            text_content=markdown_content,  # Use full markdown
                            attributes=attrs,
                        )
                    )
            # --- End Populate Match Details --- #

            # --- Perform size validation if unique element found --- #
            if count == 1 and max_html_length is not None and isinstance(elements[0], Tag):
                try:
                    html_content = str(elements[0])
                    # Use DEFAULT_MAX_HTML_LENGTH_VALIDATION if max_html_length is passed as None explicitly by caller (though signature has int)
                    # or just use the provided max_html_length if it's a valid int.
                    effective_max_html_length = (
                        max_html_length
                        if isinstance(max_html_length, int)
                        else DEFAULT_MAX_HTML_LENGTH_VALIDATION
                    )

                    html_len = len(html_content)
                    if html_len > effective_max_html_length:
                        size_validation_error_msg = f"Element HTML too large: {html_len} chars > {effective_max_html_length}"
                        logger.warning(f"{log_prefix}: {size_validation_error_msg}")
                    else:
                        logger.debug(
                            f"{log_prefix}: Size validation OK ({html_len} chars <= {effective_max_html_length})"
                        )
                except Exception as size_err:
                    size_validation_error_msg = f"Error during size validation: {size_err}"
                    logger.warning(f"{log_prefix}: {size_validation_error_msg}")
            # --- End size validation --- #

            # --- Generate feedback if selector is not unique --- #
            feedback = None
            if count > 1:
                feedback = f"Selector matched {count} elements. Review the 'matches' list for details (tag, attributes, markdown content) of the first {len(match_details)} matches to help refine your selector."

            # --- Check for potential selector over-simplicity --- #
            simplicity_warning = None
            # Constants for simplicity check
            max_markdown_len_threshold = 3000
            common_structural_tags = {"body", "html", "main", "article", "header", "footer", "nav"}
            # Check only if we found at least one element
            if count > 0 and isinstance(elements[0], Tag):
                first_el = elements[0]
                # Check 1: Is the selector just a simple, generic tag name?
                is_simple_tag_selector = (
                    selector.strip().isalnum()
                    and selector.strip().lower() not in common_structural_tags
                    and len(selector.strip().split()) == 1  # Ensure no complex attributes etc.
                )
                if is_simple_tag_selector:
                    simplicity_warning = f"Warning: Selector ('{selector}') uses only a generic tag name. Consider adding classes, IDs, or attributes for stability and specificity."
                # Check 2: Is the content length excessive? (Only if no simple tag warning yet)
                if not simplicity_warning:
                    # Re-use markdown if already calculated, else calculate it now
                    markdown_content = (
                        markdown_content
                        if "markdown_content" in locals()
                        else self._convert_html_to_markdown(first_el)
                    )
                    markdown_len = len(markdown_content)
                    if markdown_len > max_markdown_len_threshold:
                        simplicity_warning = f"Warning: Selector matched an element with very large content ({markdown_len} chars markdown > {max_markdown_len_threshold}). Consider refining for more specificity."
                        logger.warning(f"{log_prefix}: {simplicity_warning}")
            # --- End Simplicity Check --- #

            result = SelectorEvaluationResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_count=count,
                matches=match_details,  # Now contains richer details
                target_text_found_in_any_match=text_found_flag,
                error=None,
                size_validation_error=size_validation_error_msg,
                feedback_message=feedback,  # Updated feedback message
                simplicity_warning=simplicity_warning,  # Added simplicity warning
                matched_html_snippets=matched_html if return_matched_html else None,
            )
            logger.info(
                f"{log_prefix}: Result: Count={result.element_count}, TextFound={result.target_text_found_in_any_match}, MatchesDetailed={len(result.matches)}"
            )
            # Slightly more detailed log for the first match if present
            if result.matches:
                first = result.matches[0]
                # Log truncated markdown for brevity in this specific log line
                log_md_preview = (
                    (first.text_content[:100] + "...")
                    if first.text_content and len(first.text_content) > 100
                    else first.text_content
                )
                logger.debug(
                    f"{log_prefix}: First Match: <{first.tag_name}> attrs={first.attributes} markdown='{log_md_preview}'"
                )
            return result
        except Exception as e:
            error_msg = f"Evaluation Error: {type(e).__name__}: {e}"
            tb_str = traceback.format_exc()
            logger.error(f"{log_prefix}: {error_msg}", exc_info=False)  # No traceback in log
            # TRUNCATE Traceback for the agent to avoid excessive length
            max_tb_len = 1000
            if len(tb_str) > max_tb_len:
                tb_str = tb_str[:max_tb_len] + "\n... (Traceback truncated)"
            error_for_agent = f"{error_msg}\nTraceback:\n{tb_str}"
            return SelectorEvaluationResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_count=0,
                error=error_for_agent,  # Pass traceback to agent
                matches=[],
                target_text_found_in_any_match=False,
                size_validation_error=None,
                feedback_message=None,
                matched_html_snippets=None,
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
        base_element: BeautifulSoup | Tag | None = self.soup
        parent_element: Optional[Tag] = None
        if anchor_selector:
            try:
                possible_anchors = self.soup.select(anchor_selector)
            except Exception as e:
                error_msg = f"Anchor Selector Syntax Error: {type(e).__name__}: {e}"
                logger.warning(f"{log_prefix}: {error_msg}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    error=error_msg,
                    children_details=None,
                )

            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    error=error_msg,
                    children_details=None,
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)})."
                logger.warning(f"{log_prefix}: {error_msg}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    error=error_msg,
                    children_details=None,
                )
            base_element = possible_anchors[0]
            logger.debug(f"{log_prefix}: Anchor found successfully.")

        assert base_element is not None, "Base element for get_children_tags cannot be None"

        max_snippet_len = DEFAULT_MAX_SNIPPET_LENGTH  # Use constant
        try:
            parent_element = base_element.select_one(selector)
            if parent_element and isinstance(parent_element, Tag):
                details_list: list[ChildDetail] = []
                children_count = 0
                # --- Limit number of children detailed ---
                for child in parent_element.find_all(recursive=False):
                    if isinstance(child, Tag) and child.name:
                        children_count += 1
                        if children_count <= DEFAULT_MAX_CHILDREN_TO_DETAIL:  # Check limit
                            snippet = str(child)
                            if len(snippet) > max_snippet_len:
                                snippet = snippet[:max_snippet_len] + "..."
                            details_list.append(
                                ChildDetail(tag_name=child.name, html_snippet=snippet)
                            )

                # Log total count vs detailed count
                child_tags_summary = (
                    f"{len(details_list)} detailed out of {children_count} total"
                    if children_count > 0
                    else "None"
                )

                logger.info(
                    f"{log_prefix}: Result: ParentFound=True, ChildrenTags Summary=[{child_tags_summary}]"
                )
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=True,
                    children_details=details_list,
                    error=None,
                )
            else:
                error_msg = "Parent selector did not match any element or matched a non-Tag within the specified context."
                logger.info(f"{log_prefix}: Result: ParentFound=False. {error_msg}")
                return ChildrenTagsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    parent_found=False,
                    error=error_msg,
                    children_details=None,
                )
        except Exception as e:
            error_msg = f"Error getting children details: {type(e).__name__}: {e}"
            tb_str = traceback.format_exc()
            logger.error(f"{log_prefix}: {error_msg}", exc_info=False)  # No traceback in log
            # TRUNCATE Traceback for the agent
            max_tb_len = 1000
            if len(tb_str) > max_tb_len:
                tb_str = tb_str[:max_tb_len] + "\n... (Traceback truncated)"
            error_for_agent = f"{error_msg}\nTraceback:\n{tb_str}"
            return ChildrenTagsResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                parent_found=False,
                error=error_for_agent,  # Pass traceback to agent
                children_details=None,
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
        base_element: BeautifulSoup | Tag | None = self.soup
        element: Optional[Tag] = None
        if anchor_selector:
            try:
                possible_anchors = self.soup.select(anchor_selector)
            except Exception as e:
                error_msg = f"Anchor Selector Syntax Error: {type(e).__name__}: {e}"
                logger.warning(f"{log_prefix}: {error_msg}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    error=error_msg,
                    siblings=[],
                )

            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    error=error_msg,
                    siblings=[],
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)})."
                logger.warning(f"{log_prefix}: {error_msg}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    error=error_msg,
                    siblings=[],
                )
            base_element = possible_anchors[0]
            logger.debug(f"{log_prefix}: Anchor found successfully.")

        assert base_element is not None, "Base element for sibling search cannot be None"

        try:
            element = base_element.select_one(selector)
            if not element or not isinstance(element, Tag):
                error_msg = "Selector did not match any element or matched a non-Tag within the specified context."
                logger.info(f"{log_prefix}: Result: ElementFound=False. {error_msg}")
                return SiblingsResult(
                    selector_used=selector,
                    anchor_selector_used=anchor_selector,
                    element_found=False,
                    error=error_msg,
                    siblings=[],
                )

            siblings_details: list[SiblingDetail] = []
            logger.debug(f"{log_prefix}: Reference element found: <{element.name}>")

            siblings_summary_list = []
            # Previous sibling - ensuring it's a Tag
            prev_sib = element.find_previous_sibling()
            while prev_sib and not isinstance(prev_sib, Tag):
                prev_sib = prev_sib.find_previous_sibling()  # Skip non-Tag nodes
            if prev_sib and isinstance(prev_sib, Tag):
                attrs = {
                    k: " ".join(v) if isinstance(v, list) else v for k, v in prev_sib.attrs.items()
                }
                siblings_details.append(
                    SiblingDetail(tag_name=prev_sib.name, direction="previous", attributes=attrs)
                )
                logger.debug(
                    f"{log_prefix}: Found Previous Sibling: <{prev_sib.name}> attrs={attrs}"
                )
                siblings_summary_list.append(f"prev=<{prev_sib.name}>")

            # Next sibling - ensuring it's a Tag
            next_sib = element.find_next_sibling()
            while next_sib and not isinstance(next_sib, Tag):
                next_sib = next_sib.find_next_sibling()  # Skip non-Tag nodes
            if next_sib and isinstance(next_sib, Tag):
                attrs = {
                    k: " ".join(v) if isinstance(v, list) else v for k, v in next_sib.attrs.items()
                }
                siblings_details.append(
                    SiblingDetail(tag_name=next_sib.name, direction="next", attributes=attrs)
                )
                logger.debug(f"{log_prefix}: Found Next Sibling: <{next_sib.name}> attrs={attrs}")
                siblings_summary_list.append(f"next=<{next_sib.name}>")

            siblings_summary = ", ".join(siblings_summary_list) if siblings_summary_list else "None"
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
            tb_str = traceback.format_exc()
            logger.error(f"{log_prefix}: {error_msg}", exc_info=False)  # No traceback in log
            # TRUNCATE Traceback for the agent
            max_tb_len = 1000
            if len(tb_str) > max_tb_len:
                tb_str = tb_str[:max_tb_len] + "\n... (Traceback truncated)"
            error_for_agent = f"{error_msg}\nTraceback:\n{tb_str}"
            return SiblingsResult(
                selector_used=selector,
                anchor_selector_used=anchor_selector,
                element_found=False,
                error=error_for_agent,  # Pass traceback to agent
                siblings=[],
            )

    async def extract_data_from_element(
        self,
        selector: str,
        attribute_to_extract: Optional[str] = None,
        extract_text: bool = False,
        anchor_selector: Optional[str] = None,
    ) -> ExtractionResult:
        """Extracts data (attribute or text, plus HTML/Markdown) from the FIRST element matching the selector (optionally within anchor). Assumes selector is unique."""
        log_prefix = (
            f"Extract Data ('{selector}'"
            + (f" within '{anchor_selector}'" if anchor_selector else "")
            + f", attr='{attribute_to_extract}', text={extract_text})"
        )
        logger.info(f"{log_prefix}: Starting extraction.")

        base_element: BeautifulSoup | Tag | None = self.soup
        element: Optional[Tag] = None
        if anchor_selector:
            try:
                possible_anchors = self.soup.select(anchor_selector)
            except Exception as e:
                error_msg = f"Anchor Selector Syntax Error: {type(e).__name__}: {e}"
                logger.warning(f"{log_prefix}: {error_msg}")
                return ExtractionResult(
                    error=error_msg,
                    extracted_text=None,
                    extracted_attribute_value=None,
                    extracted_markdown=None,
                    extracted_html=None,
                )

            if len(possible_anchors) == 0:
                error_msg = (
                    f"Anchor Error: Anchor selector '{anchor_selector}' did not find any element."
                )
                logger.warning(f"{log_prefix}: {error_msg}")
                return ExtractionResult(
                    error=error_msg,
                    extracted_text=None,
                    extracted_attribute_value=None,
                    extracted_markdown=None,
                    extracted_html=None,
                )
            if len(possible_anchors) > 1:
                error_msg = f"Anchor Error: Anchor selector '{anchor_selector}' is not unique (found {len(possible_anchors)})."
                logger.warning(f"{log_prefix}: {error_msg}")
                return ExtractionResult(
                    error=error_msg,
                    extracted_text=None,
                    extracted_attribute_value=None,
                    extracted_markdown=None,
                    extracted_html=None,
                )
            base_element = possible_anchors[0]

        assert base_element is not None, "Base element for extraction cannot be None"

        try:
            element = base_element.select_one(selector)
            if not element or not isinstance(element, Tag):
                error_msg = "Extraction Error: Selector did not match any element or matched non-Tag within the specified context."
                logger.warning(f"{log_prefix}: {error_msg}")
                return ExtractionResult(
                    error=error_msg,
                    extracted_text=None,
                    extracted_attribute_value=None,
                    extracted_markdown=None,
                    extracted_html=None,
                )

            extracted_text_val: Optional[str] = None
            extracted_attr_val: Optional[str] = None
            markdown_content_val: Optional[str] = None
            html_content_val: Optional[str] = None

            # Always get HTML content string if element is found
            try:
                html_content_val = str(element)
                # --- Truncate HTML ---
                if len(html_content_val) > DEFAULT_MAX_SNIPPET_LENGTH:
                    html_preview = html_content_val[:DEFAULT_MAX_SNIPPET_LENGTH] + "..."
                    logger.debug(
                        f"{log_prefix}: Extracted HTML content (truncated): '{html_preview}'"
                    )
                    html_content_val = html_preview  # Return truncated value
                else:
                    logger.debug(
                        f"{log_prefix}: Extracted HTML content: '{html_content_val[:100]}...'"
                    )
            except Exception as html_err:
                logger.warning(f"{log_prefix}: Failed to get HTML string: {html_err}")
                html_content_val = f"Error getting HTML: {html_err}"

            if extract_text:
                extracted_text_val = element.get_text(separator=" ", strip=True)
                logger.info(
                    f"{log_prefix}: Extracted text: '{extracted_text_val[:100]}...'"
                    if extracted_text_val
                    else "(No text extracted)"
                )
                # --- Pre-process links for markdown --- #
                element_copy = copy.copy(element)
                links_processed_count = 0
                for link_tag in element_copy.find_all("a", href=True):
                    # --- Ensure it's a Tag before accessing attributes --- #
                    if not isinstance(link_tag, Tag):
                        continue
                    original_href = link_tag.get("href")  # Use .get() for safety
                    # Only process strings and if base_url exists
                    if isinstance(original_href, str) and self.base_url:
                        absolute_href = urljoin(self.base_url, original_href)
                        if absolute_href != original_href:
                            link_tag["href"] = (
                                absolute_href  # Modify the href on the copy (safe now due to isinstance check)
                            )
                            links_processed_count += 1
                if links_processed_count > 0:
                    logger.debug(
                        f"{log_prefix}: Absolutified {links_processed_count} href(s) in element copy before markdown conversion."
                    )
                # --- End Pre-process --- #

                try:
                    # Use the modified copy for markdown conversion
                    markdown_content_val = markdownify(str(element_copy), base_url=self.base_url)
                    if len(markdown_content_val) > DEFAULT_MAX_SNIPPET_LENGTH:
                        markdown_content_val = (
                            markdown_content_val[:DEFAULT_MAX_SNIPPET_LENGTH] + "..."
                        )
                        logger.debug(f"{log_prefix}: Generated truncated markdown content.")
                except Exception as md_err:
                    logger.warning(f"{log_prefix}: Failed to generate markdown content: {md_err}")
                    markdown_content_val = f"Error generating markdown: {md_err}"

            if attribute_to_extract:
                attr_value = element.get(attribute_to_extract)
                if attr_value is not None:
                    if isinstance(attr_value, list):
                        extracted_attr_val = " ".join(attr_value)
                    else:
                        extracted_attr_val = str(attr_value)
                    logger.info(
                        f"{log_prefix}: Extracted attribute '{attribute_to_extract}': '{extracted_attr_val}'"
                    )
                    # Absolutify URL if applicable
                    if (
                        attribute_to_extract in ["href", "src"]
                        and extracted_attr_val
                        and self.base_url
                    ):
                        original_val = extracted_attr_val
                        extracted_attr_val = urljoin(self.base_url, extracted_attr_val)
                        if original_val != extracted_attr_val:
                            logger.info(
                                f"{log_prefix}: Absolutified URL from '{original_val}' to '{extracted_attr_val}' using base '{self.base_url}'"
                            )
                        else:
                            logger.debug(
                                f"{log_prefix}: URL '{original_val}' was already absolute."
                            )
                else:
                    logger.warning(
                        f"{log_prefix}: Attribute '{attribute_to_extract}' not found on the element."
                    )
                    # We return None for the value, not an error in this case.

            # --- Generate Markdown AFTER potential text/attribute extraction --- #
            # We generate markdown last, using the final element state (links might be absolute)
            try:
                # Use a copy to ensure any link absolutification doesn't affect other steps if run differently
                element_copy_for_md = copy.copy(element)
                # Process links again just in case they weren't processed (e.g., if only attr was extracted)
                links_processed_count = 0
                for link_tag in element_copy_for_md.find_all("a", href=True):
                    if not isinstance(link_tag, Tag):
                        continue
                    original_href = link_tag.get("href")
                    if isinstance(original_href, str) and self.base_url:
                        absolute_href = urljoin(self.base_url, original_href)
                        if absolute_href != original_href:
                            link_tag["href"] = absolute_href
                            links_processed_count += 1
                if links_processed_count > 0:
                    logger.debug(
                        f"{log_prefix}: Absolutified {links_processed_count} href(s) before final markdown conversion."
                    )

                # --- Convert to Markdown and Truncate (again, ensure it happens regardless of path) ---
                markdown_content_val = markdownify(str(element_copy_for_md), base_url=self.base_url)
                if len(markdown_content_val) > DEFAULT_MAX_SNIPPET_LENGTH:
                    markdown_content_val = markdown_content_val[:DEFAULT_MAX_SNIPPET_LENGTH] + "..."
                    logger.debug(f"{log_prefix}: Final markdown content generated and truncated.")
            except Exception as md_err:
                logger.warning(f"{log_prefix}: Failed to generate markdown content: {md_err}")
                markdown_content_val = f"Error generating markdown: {md_err}"
            # --- End Markdown Generation --- #

            return ExtractionResult(
                error=None,
                extracted_text=extracted_text_val,
                extracted_attribute_value=extracted_attr_val,
                extracted_markdown=markdown_content_val,
                extracted_html=html_content_val,
            )

        except Exception as e:
            error_msg = f"Extraction Exception: {type(e).__name__}: {e}"
            tb_str = traceback.format_exc()
            logger.error(f"{log_prefix}: {error_msg}", exc_info=False)  # No traceback in log
            # TRUNCATE Traceback for the agent
            max_tb_len = 1000
            if len(tb_str) > max_tb_len:
                tb_str = tb_str[:max_tb_len] + "\n... (Traceback truncated)"
            error_for_agent = f"{error_msg}\nTraceback:\n{tb_str}"
            return ExtractionResult(
                error=error_for_agent,
                extracted_text=None,
                extracted_attribute_value=None,
                extracted_markdown=None,
                extracted_html=None,
            )

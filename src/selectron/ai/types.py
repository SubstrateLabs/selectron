from typing import Literal, Optional

from pydantic import BaseModel, Field


class MatchDetail(BaseModel):
    tag_name: str = Field(..., description="Tag name of the matched element.")
    text_content: Optional[str] = Field(
        None, description="Full markdown content of the matched element and its children."
    )
    attributes: dict[str, str] = Field({}, description="Dictionary of the element's attributes.")


class SelectorEvaluationResult(BaseModel):
    """Result of evaluating a CSS selector."""

    selector_used: str
    anchor_selector_used: Optional[str] = None
    element_count: int
    matches: list[MatchDetail] = Field(default_factory=list)
    target_text_found_in_any_match: bool
    error: Optional[str] = None
    size_validation_error: Optional[str] = None
    feedback_message: Optional[str] = None  # General feedback, e.g., if not unique
    simplicity_warning: Optional[str] = None  # Specific warning if selector seems too broad
    matched_html_snippets: Optional[list[str]] = None


class ChildDetail(BaseModel):
    tag_name: str = Field(...)
    html_snippet: str = Field(...)


class ChildrenTagsResult(BaseModel):
    selector_used: str = Field(...)
    anchor_selector_used: Optional[str] = Field(None)
    parent_found: bool = Field(...)
    children_details: Optional[list[ChildDetail]] = Field(None)
    error: Optional[str] = Field(None)


class SiblingDetail(BaseModel):
    tag_name: str = Field(...)
    direction: Literal["previous", "next"] = Field(...)
    attributes: dict[str, str] = Field({})


class SiblingsResult(BaseModel):
    selector_used: str = Field(...)
    anchor_selector_used: Optional[str] = Field(None)
    element_found: bool = Field(...)
    siblings: list[SiblingDetail] = Field([])
    error: Optional[str] = Field(None)


class ExtractionResult(BaseModel):
    extracted_text: Optional[str] = Field(None)
    extracted_attribute_value: Optional[str] = Field(None)
    extracted_markdown: Optional[str] = Field(
        None, description="Markdown representation of the extracted element's content."
    )
    extracted_html: Optional[str] = Field(
        None, description="Raw HTML string of the extracted element and its descendants."
    )
    error: Optional[str] = Field(None)


class AgentResult(BaseModel):
    """Structured result containing the proposed selector, reasoning, extraction details, and verification."""

    proposed_selector: str = Field(...)
    reasoning: str = Field(...)
    attribute_extracted: Optional[str] = Field(None)
    text_extracted_flag: bool = Field(False)
    extraction_result: ExtractionResult = Field(...)
    final_verification: SelectorEvaluationResult = Field(...)


class SelectorProposal(BaseModel):
    proposed_selector: str = Field(...)
    reasoning: str = Field(...)
    target_cardinality: Literal["unique", "multiple"] = Field(
        "unique",
        description="Indicates if the selector targets a single element or intentionally multiple elements.",
    )


class AutoProposal(BaseModel):
    proposed_description: str = Field(
        description="A concise natural language description of the target element(s)"
    )

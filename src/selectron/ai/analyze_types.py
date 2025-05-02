from typing import List, Optional

from pydantic import BaseModel, Field


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


class ExtractionProposal(BaseModel):
    items: List[ProposedContentRegion] = Field(
        ..., description="List of proposed content regions identified in the webpage screenshot."
    )


class MarkdownMappingItem(BaseModel):
    markdown_snippet: Optional[str] = Field(
        None,
        description="The corresponding markdown snippet for the region, or null if none found.",
    )
    metadata: Optional[dict[str, str]] = Field(
        None, description="Extracted factual metadata dictionary, or null/empty if none found."
    )


class MarkdownMappingResponse(BaseModel):
    mapped_items: List[MarkdownMappingItem] = Field(
        ...,
        description="A list containing exactly one item for each input region, in the original order. Each item bundles the markdown snippet and metadata for that region.",
    )

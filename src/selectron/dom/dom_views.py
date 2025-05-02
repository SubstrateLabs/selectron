from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional

# Import from history_tree_views
from selectron.dom.history_tree_views import (
    CoordinateSet,
    HashedDomElement,
    ViewportInfo,
)

# Avoid circular import issues
if TYPE_CHECKING:
    from .dom_views import DOMElementNode


@dataclass(frozen=False)
class DOMBaseNode:
    is_visible: bool
    # Use None as default and set parent later to avoid circular reference issues
    parent: Optional["DOMElementNode"]

    def __json__(self) -> dict:
        raise NotImplementedError("DOMBaseNode is an abstract class")


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
    text: str
    type: str = "TEXT_NODE"

    def has_parent_with_highlight_index(self) -> bool:
        current = self.parent
        while current is not None:
            # stop if the element has a highlight index (will be handled separately)
            if current.highlight_index is not None:
                return True

            current = current.parent
        return False

    def is_parent_in_viewport(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.is_in_viewport

    def is_parent_top_element(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.is_top_element

    def __json__(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
        }


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
    """
    xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
    To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
    """

    tag_name: str
    xpath: str
    attributes: Dict[str, str]
    children: List[DOMBaseNode]
    is_interactive: bool = False
    is_top_element: bool = False
    is_in_viewport: bool = False
    shadow_root: bool = False
    highlight_index: Optional[int] = None
    viewport_coordinates: Optional[CoordinateSet] = None
    page_coordinates: Optional[CoordinateSet] = None
    viewport_info: Optional[ViewportInfo] = None
    is_content_element: bool = False

    """
	### State injected by the browser context.

	The idea is that the clickable elements are sometimes persistent from the previous page -> tells the model which objects are new/_how_ the state has changed
	"""
    is_new: Optional[bool] = None

    def __json__(self) -> dict:
        return {
            "tag_name": self.tag_name,
            "xpath": self.xpath,
            "attributes": self.attributes,
            "is_visible": self.is_visible,
            "is_interactive": self.is_interactive,
            "is_top_element": self.is_top_element,
            "is_in_viewport": self.is_in_viewport,
            "shadow_root": self.shadow_root,
            "highlight_index": self.highlight_index,
            "viewport_coordinates": self.viewport_coordinates,
            "page_coordinates": self.page_coordinates,
            "children": [child.__json__() for child in self.children],
        }

    def __repr__(self) -> str:
        tag_str = f"<{self.tag_name}"

        # Add attributes
        for key, value in self.attributes.items():
            tag_str += f' {key}="{value}"'
        tag_str += ">"

        # Add extra info
        extras = []
        if self.is_interactive:
            extras.append("interactive")
        if self.is_top_element:
            extras.append("top")
        if self.shadow_root:
            extras.append("shadow-root")
        if self.highlight_index is not None:
            extras.append(f"highlight:{self.highlight_index}")
        if self.is_in_viewport:
            extras.append("in-viewport")

        if extras:
            tag_str += f" [{', '.join(extras)}]"

        return tag_str

    @cached_property
    def hash(self) -> HashedDomElement:
        from selectron.dom.history_tree_processor import HistoryTreeProcessor

        return HistoryTreeProcessor._hash_dom_element(self)

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        text_parts = []

        def collect_text(node: DOMBaseNode, current_depth: int) -> None:
            if max_depth != -1 and current_depth > max_depth:
                return

            # Skip this branch if we hit a highlighted element (except for the current node)
            if (
                isinstance(node, DOMElementNode)
                and node != self
                and node.highlight_index is not None
            ):
                return

            if isinstance(node, DOMTextNode):
                text_parts.append(node.text)
            elif isinstance(node, DOMElementNode):
                for child in node.children:
                    collect_text(child, current_depth + 1)

        collect_text(self, 0)
        return "\n".join(text_parts).strip()

    def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Convert the processed DOM content to HTML."""
        formatted_text = []

        def process_node(node: DOMBaseNode, depth: int) -> None:
            next_depth = int(depth)
            depth_str = depth * "\t"

            if isinstance(node, DOMElementNode):
                # --- Render Interactive Element ---
                if node.highlight_index is not None:
                    next_depth += 1

                    text = node.get_all_text_till_next_clickable_element()
                    attributes_html_str = ""
                    attributes_to_include = {}  # Initialize empty dict

                    # Populate attributes_to_include based on the include_attributes list
                    if include_attributes:
                        for attr_name in include_attributes:
                            if attr_name in node.attributes:
                                attributes_to_include[attr_name] = str(node.attributes[attr_name])

                        # Easy LLM optimizations (apply AFTER selecting attributes)
                        # if tag == role attribute, don't include it
                        if node.tag_name == attributes_to_include.get("role"):
                            del attributes_to_include["role"]

                        # if aria-label == text of the node, don't include it
                        # Strip both for comparison
                        aria_label_val = attributes_to_include.get("aria-label", "").strip()
                        text_val = text.strip()
                        # Check if aria_label_val is non-empty before comparison
                        if aria_label_val and aria_label_val == text_val:
                            del attributes_to_include["aria-label"]

                        # if placeholder == text of the node, don't include it
                        # Strip both for comparison
                        placeholder_val = attributes_to_include.get("placeholder", "").strip()
                        # Check if placeholder_val is non-empty before comparison
                        if placeholder_val and placeholder_val == text_val:
                            del attributes_to_include["placeholder"]

                    # Format the final selected attributes
                    if attributes_to_include:
                        # Properly escape quotes within attribute values for safe inclusion
                        # Build parts separately to avoid Python < 3.12 f-string quote limitations
                        attr_parts = []
                        for key, value in attributes_to_include.items():
                            # Escape single quotes within the value
                            escaped_value = value.replace("'", "\\'")
                            attr_parts.append(f"{key}='{escaped_value}'")
                        attributes_html_str = " ".join(attr_parts)

                    # Build the line for interactive element
                    if node.is_new:
                        highlight_indicator = f"*[{node.highlight_index}]*"
                    else:
                        highlight_indicator = f"[{node.highlight_index}]"

                    line = f"{depth_str}{highlight_indicator}<{node.tag_name}"

                    if attributes_html_str:
                        line += f" {attributes_html_str}"

                    # Simplified logic for adding text/closing tag
                    if text:
                        # Add space only if attributes were present
                        if attributes_html_str:
                            line += f" >{text} />"
                        else:
                            line += f">{text} />"  # No space needed if no attributes
                    else:
                        # Add space only if attributes were present
                        if attributes_html_str:
                            line += " />"
                        else:
                            # No space needed if no attributes
                            line += (
                                " />"  # Reverting to add space for consistency with previous code
                            )
                    formatted_text.append(line)
                    # Process children of interactive elements
                    for child in node.children:
                        process_node(child, next_depth)

                # --- Render Content Element (or structural parent without highlight) ---
                elif node.is_content_element or not any(
                    isinstance(c, DOMElementNode) and c.highlight_index is not None
                    for c in node.children
                ):  # Include structural parents only if they lead to content
                    # Only render if it has text or leads to something renderable
                    text = (
                        node.get_all_text_till_next_clickable_element()
                    )  # Maybe limit depth here?
                    # Or just get immediate text? Let's try immediate text first for content elements.
                    immediate_text = "".join(
                        c.text for c in node.children if isinstance(c, DOMTextNode)
                    ).strip()

                    # Decide whether to render this node based on text or if children will be rendered
                    has_renderable_children = any(
                        (
                            isinstance(child, DOMElementNode)
                            and (child.highlight_index is not None or child.is_content_element)
                        )
                        or (
                            isinstance(child, DOMTextNode)
                            and child.text.strip()
                            and not child.has_parent_with_highlight_index()
                        )
                        for child in node.children
                    )

                    if immediate_text or has_renderable_children:
                        # Use a different prefix/indentation? Maybe just indent without index.
                        line_prefix = f"{depth_str}  "
                        line = f"{line_prefix}<{node.tag_name}"

                        # Optionally add some attributes for context (e.g., class, id)?
                        attributes_html_str = ""
                        context_attributes = {}
                        if "class" in node.attributes and node.attributes["class"]:
                            context_attributes["class"] = node.attributes["class"]
                        if "id" in node.attributes and node.attributes["id"]:
                            context_attributes["id"] = node.attributes["id"]

                        if context_attributes:
                            attr_parts = []
                            for key, value in context_attributes.items():
                                escaped_value = str(value).replace("'", "\\'")
                                attr_parts.append(f"{key}='{escaped_value}'")
                            attributes_html_str = " ".join(attr_parts)
                            line += f" {attributes_html_str}"

                        # Add immediate text if present
                        if immediate_text:
                            if attributes_html_str:
                                line += " "  # Add space if attributes were added
                            line += f">{immediate_text} />"
                        else:
                            if attributes_html_str:
                                line += " "  # Add space if attributes were added
                            line += "/>"  # Self-close if no text

                        formatted_text.append(line)

                        # Process children for content elements too
                        for child in node.children:
                            process_node(
                                child, depth + 1
                            )  # Keep same depth or increment? Let's increment.
                    else:
                        # If it's a structural node with no immediate text and no renderable children,
                        # still process its children at the *same* depth, don't render the parent itself.
                        for child in node.children:
                            process_node(child, depth)
                else:
                    # If it's neither highlighted nor content/structural, just process children at same depth.
                    for child in node.children:
                        process_node(child, depth)

            elif isinstance(node, DOMTextNode):
                # Add text only if it doesn't have a highlighted parent AND is relevant
                if (
                    not node.has_parent_with_highlight_index()
                    and node.parent
                    and node.parent.is_visible
                    and node.parent.is_top_element
                    # Also check if parent is a content element we decided to include
                    # (This logic might need refinement based on how structural parents are handled above)
                    # and (node.parent.highlight_index is not None or node.parent.is_content_element)
                ):
                    # Indent text relative to its parent element
                    formatted_text.append(
                        f"{depth_str}  {node.text.strip()}"
                    )  # Add extra indent for text

        process_node(self, 0)
        return "\n".join(formatted_text)

    def get_file_upload_element(self, check_siblings: bool = True) -> Optional["DOMElementNode"]:
        # Check if current element is a file input
        if self.tag_name == "input" and self.attributes.get("type") == "file":
            return self

        # Check children
        for child in self.children:
            if isinstance(child, DOMElementNode):
                result = child.get_file_upload_element(check_siblings=False)
                if result:
                    return result

        # Check siblings only for the initial call
        if check_siblings and self.parent:
            for sibling in self.parent.children:
                if sibling is not self and isinstance(sibling, DOMElementNode):
                    result = sibling.get_file_upload_element(check_siblings=False)
                    if result:
                        return result

        return None


SelectorMap = dict[int, DOMElementNode]


@dataclass
class DOMState:
    element_tree: DOMElementNode
    selector_map: SelectorMap

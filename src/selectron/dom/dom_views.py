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

    def elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Convert the processed DOM content to a simplified string representation."""
        formatted_text = []

        def process_node(node: DOMBaseNode, depth: int) -> None:
            if not node.is_visible:  # Skip invisible nodes entirely
                return

            depth_str = depth * "\t"

            if isinstance(node, DOMElementNode):
                tag = node.tag_name
                attrs_str = ""
                attributes_to_include = {}

                # Basic attributes for all visible elements for context
                basic_attrs = [
                    "id",
                    "class",
                    "role",
                    "name",
                    "data-testid",
                    "aria-label",
                    "placeholder",
                    "title",
                    "alt",
                    "href",
                    "type",
                    "for",
                ]
                for attr_name in basic_attrs:
                    if attr_name in node.attributes and node.attributes[attr_name]:
                        attributes_to_include[attr_name] = str(node.attributes[attr_name])

                # If it's highlighted, use the more specific attribute logic
                if node.highlight_index is not None:
                    if include_attributes:
                        # Start fresh for highlighted, apply include_attributes logic
                        attributes_to_include = {}
                        for attr_name in include_attributes:
                            if attr_name in node.attributes:
                                attributes_to_include[attr_name] = str(node.attributes[attr_name])

                        # Apply optimizations
                        if node.tag_name == attributes_to_include.get("role"):
                            del attributes_to_include["role"]
                        aria_label_val = attributes_to_include.get("aria-label", "").strip()
                        # Placeholder for text extraction (simplified for now)
                        # We might need a simpler text getter if get_all_text_till_next_clickable_element is too complex/buggy
                        # For now, let's assume direct text children are most important
                        immediate_text = "".join(
                            c.text
                            for c in node.children
                            if isinstance(c, DOMTextNode) and c.is_visible
                        ).strip()
                        if aria_label_val and aria_label_val == immediate_text:
                            del attributes_to_include["aria-label"]
                        placeholder_val = attributes_to_include.get("placeholder", "").strip()
                        if placeholder_val and placeholder_val == immediate_text:
                            del attributes_to_include["placeholder"]

                # Format the final attributes string
                if attributes_to_include:
                    attr_parts = []
                    for key, value in attributes_to_include.items():
                        escaped_value = value.replace("'", "\\'")  # Ensure proper escaping
                        attr_parts.append(f"{key}='{escaped_value}'")
                    attrs_str = " ".join(attr_parts)

                # --- Build the output line ---
                highlight_indicator = ""
                if node.highlight_index is not None:
                    highlight_indicator = f"[{node.highlight_index}]"
                    if node.is_new:
                        highlight_indicator = (
                            f"*{highlight_indicator}*"  # Add star for new elements
                        )

                line = f"{depth_str}{highlight_indicator}<{tag}"
                if attrs_str:
                    line += f" {attrs_str}"
                line += " />"  # Always self-close for this simplified representation for now
                formatted_text.append(line)

                # Always recurse for children of visible elements, incrementing depth
                for child in node.children:
                    process_node(child, depth + 1)

            elif isinstance(node, DOMTextNode):
                # Render visible text nodes
                text = node.text.strip()
                if text:  # Only add if there's actual text content
                    # Add text indented under its parent
                    formatted_text.append(f"{depth_str}  {text}")  # Indent text further

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

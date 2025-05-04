import logging
from importlib import resources
from typing import Optional, Protocol

from selectron.dom.dom_views import (
    DOMBaseNode,
    DOMElementNode,
    DOMState,
    DOMTextNode,
    SelectorMap,
)
from selectron.dom.history_tree_views import ViewportInfo

logger = logging.getLogger(__name__)


class BrowserExecutor(Protocol):
    async def evaluate(self, expression: str, arg: Optional[dict] = None) -> dict:
        """Evaluates JavaScript expression in the page context."""
        ...

    @property
    def url(self) -> str:
        """Gets the URL of the current page."""
        ...

    # Context manager methods for resource management (like websocket connection)
    async def __aenter__(self) -> "BrowserExecutor": ...

    async def __aexit__(self, exc_type, exc_val, exc_tb): ...


class DomService:
    def __init__(self, browser_executor: BrowserExecutor):
        self.browser_executor = browser_executor
        # self.xpath_cache = {} # This cache seems unused, consider removing later?

        self.js_code = resources.files("selectron.dom").joinpath("buildDomTree.js").read_text()

    async def get_elements(
        self,
        focus_element: int = -1,
        # viewport_expansion: int = 0, # REMOVED - Always disable viewport check
    ) -> DOMState:
        # Always use viewportExpansion=-1 to get the full DOM
        viewport_expansion = -1
        element_tree, selector_map = await self._build_dom_tree(focus_element, viewport_expansion)
        return DOMState(element_tree=element_tree, selector_map=selector_map)

    async def _build_dom_tree(
        self,
        focus_element: int,
        viewport_expansion: int,
    ) -> tuple[DOMElementNode, SelectorMap]:
        # Use browser_executor instead of page
        # Check basic evaluation works
        if await self.browser_executor.evaluate("1+1") != 2:
            raise ValueError("The browser executor cannot evaluate javascript code properly")

        # Use browser_executor.url
        if self.browser_executor.url == "about:blank":
            # short-circuit if the page is a new empty tab for speed, no need to inject buildDomTree.js
            return (
                DOMElementNode(
                    tag_name="body",
                    xpath="",
                    attributes={},
                    children=[],
                    is_visible=False,
                    parent=None,
                ),
                {},
            )

        # NOTE: We execute JS code in the browser to extract important DOM information.
        #       The returned hash map contains information about the DOM tree and the
        #       relationship between the DOM elements.
        debug_mode = logger.getEffectiveLevel() == logging.DEBUG
        args = {
            "viewportExpansion": viewport_expansion,
            "debugMode": debug_mode,
            "doHighlightElements": False,
        }

        try:
            # Use browser_executor.evaluate
            eval_page: Optional[dict] = await self.browser_executor.evaluate(self.js_code, args)
        except Exception as e:
            logger.error(
                "Error evaluating JavaScript via browser executor: %s", e, exc_info=True
            )  # Log exception info
            # Return empty structure on evaluation error
            return (
                DOMElementNode(
                    tag_name="body",
                    xpath="",
                    attributes={},
                    children=[],
                    is_visible=False,
                    parent=None,
                ),
                {},
            )

        # Check if evaluation returned None unexpectedly
        if eval_page is None:
            logger.error(
                "JavaScript evaluation via browser executor returned None unexpectedly for URL: %s. Returning empty DOM.",
                self.browser_executor.url,
            )
            return (
                DOMElementNode(
                    tag_name="body",
                    xpath="",
                    attributes={},
                    children=[],
                    is_visible=False,
                    parent=None,
                ),
                {},
            )

        return await self._construct_dom_tree(eval_page)

    async def _construct_dom_tree(
        self,
        eval_page: dict,
    ) -> tuple[DOMElementNode, SelectorMap]:
        js_node_map = eval_page["map"]
        js_root_id = eval_page["rootId"]
        # logger.debug(f"Received {len(js_node_map)} nodes from buildDomTree.js")
        selector_map = {}
        node_map = {}
        for id, node_data in js_node_map.items():
            node, children_ids = self._parse_node(node_data)
            if node is None:
                continue

            node_map[id] = node

            if isinstance(node, DOMElementNode) and node.highlight_index is not None:
                selector_map[node.highlight_index] = node

            # NOTE: We know that we are building the tree bottom up
            #       and all children are already processed.
            if isinstance(node, DOMElementNode):
                for child_id in children_ids:
                    if child_id not in node_map:
                        continue

                    child_node = node_map[child_id]

                    child_node.parent = node
                    node.children.append(child_node)

        html_to_dict = node_map[str(js_root_id)]

        del node_map
        del js_node_map
        del js_root_id

        if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):
            raise ValueError("Failed to parse HTML to dictionary")

        # Ensure the root node itself is marked as visible, as JS doesn't explicitly set it for body
        html_to_dict.is_visible = True

        return html_to_dict, selector_map

    def _parse_node(
        self,
        node_data: dict,
    ) -> tuple[Optional[DOMBaseNode], list[int]]:
        if not node_data:
            return None, []

        # Process text nodes immediately
        if node_data.get("type") == "TEXT_NODE":
            text_node = DOMTextNode(
                text=node_data["text"],
                is_visible=node_data["isVisible"],
                parent=None,
            )
            return text_node, []

        # Process coordinates if they exist for element nodes

        viewport_info = None

        if "viewport" in node_data:
            try:
                viewport_info_data = node_data["viewport"]
                viewport_info = ViewportInfo(
                    width=viewport_info_data["width"],
                    height=viewport_info_data["height"],
                    # Assuming scroll_x/y might be missing in the JS payload, add defaults or fetch differently
                    # scroll_x=viewport_info_data.get('scrollX', 0),
                    # scroll_y=viewport_info_data.get('scrollY', 0),
                )
            except KeyError as e:
                logger.warning(f"Missing key in viewport data: {e}. Skipping viewport_info.")
                viewport_info = None  # Ensure it's None if parsing fails

        element_node = DOMElementNode(
            tag_name=node_data["tagName"],
            xpath=node_data["xpath"],
            attributes=node_data.get("attributes", {}),
            children=[],
            is_visible=node_data.get("isVisible", False),
            is_interactive=node_data.get("isInteractive", False),
            is_top_element=node_data.get("isTopElement", False),
            is_in_viewport=node_data.get("isInViewport", False),
            highlight_index=node_data.get("highlightIndex"),
            shadow_root=node_data.get("shadowRoot", False),
            parent=None,
            viewport_info=viewport_info,
            is_content_element=node_data.get("isContentElement", False),
        )

        children_ids = node_data.get("children", [])

        return element_node, children_ids

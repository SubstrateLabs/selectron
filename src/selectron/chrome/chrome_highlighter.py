from typing import Optional

import websockets

from selectron.chrome.cdp_executor import CdpBrowserExecutor
from selectron.chrome.types import TabReference
from selectron.util.logger import get_logger

logger = get_logger(__name__)


class ChromeHighlighter:
    def __init__(self):
        self._highlights_active: bool = False
        self._last_highlight_selector: Optional[str] = None
        self._last_highlight_color: Optional[str] = None

    async def highlight(
        self, tab_ref: Optional[TabReference], selector: str, color: str = "yellow"
    ) -> bool:
        """Highlights elements matching a selector using overlays.

        Returns:
            bool: highlight_success
        """
        highlight_success = False

        if not tab_ref or not tab_ref.ws_url:
            logger.warning(
                "Cannot highlight selector: Missing active tab reference or websocket URL."
            )
            self._highlights_active = False
            return False

        # --- Determine alternating color logic --- #
        current_color = color
        alternate_color_map = {
            "yellow": "orange",
            "blue": "purple",
            "red": "brown",
            "lime": "green",  # Final success highlight
        }
        if color in alternate_color_map and self._last_highlight_color == color:
            current_color = alternate_color_map[color]
        # --- End alternate color logic --- #

        # --- Store state for re-highlighting ---
        self._last_highlight_selector = selector
        self._last_highlight_color = current_color
        self._highlights_active = True
        # --- End store state ---

        # --- Create Executor Once --- #
        if not tab_ref.ws_url:
            logger.error("Internal error: ws_url became None unexpectedly.")
            return False
        executor = CdpBrowserExecutor(tab_ref.ws_url, tab_ref.url or "")

        # --- Clear previous highlights FIRST (using the same executor) ---
        await self.clear(tab_ref, called_internally=True, executor=executor)
        # --- End clear previous ---

        tab_id = tab_ref.id
        # Escape the selector string for use within the JS string literal
        escaped_selector = (
            selector.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("`", "\\`")
        )

        highlight_style = f"2px solid {current_color}"
        background_color = current_color + "33"
        container_id = "selectron-highlight-container"
        overlay_attribute = "data-selectron-highlight-overlay"

        # JS Code remains the same as in cli.py
        js_code = f"""
        (function() {{
            const selector = `{escaped_selector}`;
            const borderStyle = '{highlight_style}';
            const bgColor = '{background_color}';
            const containerId = '{container_id}';
            const overlayAttr = '{overlay_attribute}';

            // Find or create the container
            let container = document.getElementById(containerId);
            if (!container) {{
                container = document.createElement('div');
                container.id = containerId;
                container.style.position = 'fixed';
                container.style.pointerEvents = 'none';
                container.style.top = '0';
                container.style.left = '0';
                container.style.width = '100%';
                container.style.height = '100%';
                container.style.zIndex = '2147483647'; // Max z-index
                container.style.backgroundColor = 'transparent';
                // Append to body if available, otherwise documentElement
                (document.body || document.documentElement).appendChild(container);
            }}

            const elements = document.querySelectorAll(selector);
            if (!elements || elements.length === 0) {{
                return `No elements found for selector: ${selector}`;
            }}

            let highlightedCount = 0;
            elements.forEach(el => {{
                try {{
                    const rects = el.getClientRects();
                    if (!rects || rects.length === 0) return; // Skip elements without geometry

                    for (const rect of rects) {{
                        if (rect.width === 0 || rect.height === 0) continue; // Skip empty rects

                        const overlay = document.createElement('div');
                        overlay.setAttribute(overlayAttr, 'true'); // Mark as overlay
                        overlay.style.position = 'fixed';
                        overlay.style.border = borderStyle;
                        overlay.style.backgroundColor = bgColor;
                        overlay.style.pointerEvents = 'none';
                        overlay.style.boxSizing = 'border-box';
                        overlay.style.top = `${{rect.top}}px`;
                        overlay.style.left = `${{rect.left}}px`;
                        overlay.style.width = `${{rect.width}}px`;
                        overlay.style.height = `${{rect.height}}px`;
                        overlay.style.zIndex = '2147483647'; // Ensure overlay is on top

                        container.appendChild(overlay);
                    }}
                    highlightedCount++;
                }} catch (e) {{
                     console.warn('Selectron highlight error for one element:', e);
                }}
            }});

            return `Highlighted ${{highlightedCount}} element(s) (using overlays) for: ${{selector}}`;
        }})();
        """

        try:
            result = await executor.evaluate(js_code)
            if result and (
                "Highlighted" in result
                or "container not found" in result
                or "Removed highlight container" in result
            ):
                highlight_success = True
            else:
                logger.warning(f"Highlight JS returned unexpected value: {result}")
                highlight_success = False

        except websockets.exceptions.WebSocketException as e:
            logger.error(f"Highlight selector failed for tab {tab_id}: WebSocket error - {e}")
            self._highlights_active = False
            return False
        except Exception as e:
            logger.error(
                f"Highlight selector failed for tab {tab_id}: Unexpected error - {e}",
                exc_info=True,
            )
            self._highlights_active = False
            return False

        return highlight_success

    async def clear(
        self,
        tab_ref: Optional[TabReference],
        called_internally: bool = False,
        executor: Optional[CdpBrowserExecutor] = None,
    ) -> None:
        """Removes all highlights previously added by this tool.

        Can use a provided executor or create a temporary one.
        """
        if not tab_ref or not tab_ref.ws_url:
            # Don't warn if called internally during highlight process
            if not called_internally:
                logger.debug(
                    "Cannot clear highlights: Missing active tab reference or websocket URL."
                )
            return

        # If not called internally (e.g., explicitly clearing), reset state
        if not called_internally:
            self._highlights_active = False
            self._last_highlight_selector = None
            self._last_highlight_color = None

        ws_url = tab_ref.ws_url
        url = tab_ref.url
        tab_id = tab_ref.id

        container_id = "selectron-highlight-container"

        # JS Code remains the same as in cli.py
        js_code = f"""
        (function() {{
            const containerId = '{container_id}'; // Capture ID for message
            const container = document.getElementById(containerId);
            let count = 0;
            if (container) {{
                count = container.childElementCount; // Count overlays before removing
                try {{
                    container.remove(); // Remove the whole container
                    return `SUCCESS: Removed highlight container ('${{containerId}}') with ${{count}} overlays.`;
                }} catch (e) {{
                    return `ERROR: Failed to remove container ('${{containerId}}'): ${{e.message}}`;
                }}
            }} else {{
                return 'INFO: Highlight container not found, nothing to remove.';
            }}
        }})();
        """

        try:
            if executor:
                await executor.evaluate(js_code)
            else:
                executor = CdpBrowserExecutor(ws_url, url or "")
                await executor.evaluate(js_code)
        except websockets.exceptions.WebSocketException:
            # Ignore connection errors during cleanup, tab might be closed
            pass
        except Exception as e:
            logger.warning(f"Non-critical error clearing highlights on tab {tab_id}: {e}")

    async def rehighlight(self, tab_ref: Optional[TabReference]):
        if not tab_ref:
            return

        if self._highlights_active and self._last_highlight_selector and self._last_highlight_color:
            selector = self._last_highlight_selector
            current_color = self._last_highlight_color  # Use the stored color
            tab_id = tab_ref.id
            ws_url = tab_ref.ws_url
            url = tab_ref.url

            if not ws_url:
                logger.warning(
                    f"[HighlightService] Cannot rehighlight on tab {tab_id}: Missing websocket URL."
                )
                return

            # --- Replicate JS execution logic from highlight() MINUS the clear() --- #
            escaped_selector = (
                selector.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace("`", "\\`")
            )
            highlight_style = f"2px solid {current_color}"
            background_color = current_color + "33"
            container_id = "selectron-highlight-container"
            overlay_attribute = "data-selectron-highlight-overlay"

            js_code = f"""
            (function() {{
                const selector = `{escaped_selector}`;
                const borderStyle = '{highlight_style}';
                const bgColor = '{background_color}';
                const containerId = '{container_id}';
                const overlayAttr = '{overlay_attribute}';

                // --- Start: Difference from highlight() --- 
                // Ensure container exists, but DO NOT clear its children first
                let container = document.getElementById(containerId);
                if (!container) {{
                    container = document.createElement('div');
                    container.id = containerId;
                    container.style.position = 'fixed';
                    container.style.pointerEvents = 'none';
                    container.style.top = '0';
                    container.style.left = '0';
                    container.style.width = '100%';
                    container.style.height = '100%';
                    container.style.zIndex = '2147483647';
                    container.style.backgroundColor = 'transparent';
                    (document.body || document.documentElement).appendChild(container);
                }} else {{
                    // If container exists, clear ONLY old overlays before drawing new ones for rehighlight
                    const oldOverlays = container.querySelectorAll(`[${{overlayAttr}}="true"]`);
                    oldOverlays.forEach(o => o.remove());
                }}
                // --- End: Difference from highlight() --- 

                const elements = document.querySelectorAll(selector);
                if (!elements || elements.length === 0) {{
                    return `Rehighlight: No elements found for selector: ${{selector}}`;
                }}
                let highlightedCount = 0;
                elements.forEach(el => {{
                    try {{
                        const rects = el.getClientRects();
                        if (!rects || rects.length === 0) return;
                        for (const rect of rects) {{
                            if (rect.width === 0 || rect.height === 0) continue;
                            const overlay = document.createElement('div');
                            overlay.setAttribute(overlayAttr, 'true');
                            overlay.style.position = 'fixed';
                            overlay.style.border = borderStyle;
                            overlay.style.backgroundColor = bgColor;
                            overlay.style.pointerEvents = 'none';
                            overlay.style.boxSizing = 'border-box';
                            overlay.style.top = `${{rect.top}}px`;
                            overlay.style.left = `${{rect.left}}px`;
                            overlay.style.width = `${{rect.width}}px`;
                            overlay.style.height = `${{rect.height}}px`;
                            overlay.style.zIndex = '2147483647';
                            container.appendChild(overlay);
                        }}
                        highlightedCount++;
                    }} catch (e) {{
                         console.warn('Selectron rehighlight error for one element:', e);
                    }}
                }});
                return `Rehighlight: Drew ${{highlightedCount}} overlays for: ${{selector}}`;
            }})();
            """
            try:
                executor = CdpBrowserExecutor(ws_url, url or "")
                await executor.evaluate(js_code)
            except websockets.exceptions.WebSocketException as e:
                logger.warning(f"Rehighlight failed for tab {tab_id}: WebSocket error - {e}")
            except Exception as e:
                logger.error(
                    f"Rehighlight failed for tab {tab_id}: Unexpected error - {e}",
                    exc_info=True,
                )
        else:
            logger.debug("Skipping rehighlight (not active or no selector/color)")

    def is_active(self) -> bool:
        """Returns true if highlights are considered active."""
        return self._highlights_active

    def set_active(self, active: bool):
        """Explicitly set the active state (e.g., when cancelling agent)."""
        self._highlights_active = active
        if not active:
            self._last_highlight_color = None
            self._last_highlight_selector = None

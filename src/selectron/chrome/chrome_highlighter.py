import json
from typing import Any, Optional

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
        self._agent_status_badge_id = "selectron-agent-status-badge"
        # additional container id for persistent parser highlight overlays
        self._parser_container_id = "selectron-parser-highlight-container"
        # tracking parser overlay
        self._parser_last_selector: Optional[str] = None
        self._parser_last_color: Optional[str] = None

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

        current_color = color
        alternate_color_map = {
            "yellow": "orange",
            "blue": "purple",
            "red": "brown",
            "lime": "green",  # Final success highlight
        }
        if color in alternate_color_map and self._last_highlight_color == color:
            current_color = alternate_color_map[color]

        self._last_highlight_selector = selector
        self._last_highlight_color = current_color
        self._highlights_active = True

        # --- Create Executor Once --- #
        temp_executor = None
        if tab_ref.ws_url:
            try:
                # Create one executor to potentially reuse for clear and highlight JS
                temp_executor = CdpBrowserExecutor(tab_ref.ws_url, tab_ref.url or "")
            except Exception as e:
                logger.error(f"Failed to create executor for highlighting: {e}")
                self._highlights_active = False
                return False
        else:
            logger.error("Highlight error: ws_url is None")
            self._highlights_active = False
            return False

        # --- Clear previous highlights FIRST (using the same executor) ---
        await self.clear(tab_ref, called_internally=True, executor=temp_executor)
        # --- End clear previous ---

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

        # Use the helper method to execute the highlight JS, passing the same executor
        result = await self._execute_js_on_tab(
            tab_ref,
            js_code,
            purpose=f"highlight selector '{selector[:30]}...'",
            executor=temp_executor,  # Reuse executor
        )

        if (
            result
            and isinstance(result, str)
            and ("Highlighted" in result or "No elements found" in result)
        ):
            highlight_success = True
        else:
            logger.warning(f"Highlight JS returned unexpected value: {result}")
            highlight_success = False
            self._highlights_active = False  # Mark as inactive on failure

        # Assuming temp_executor is managed (closed) by _execute_js_on_tab or CDP library handles closure implicitly

        return highlight_success

    async def clear(
        self,
        tab_ref: Optional[TabReference],
        called_internally: bool = False,
        executor: Optional[CdpBrowserExecutor] = None,
    ) -> None:
        """Removes all highlights previously added by this tool.

        Can use a provided executor or create a temporary one via _execute_js_on_tab.
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

        # Use helper to execute JS, passing the executor if provided
        await self._execute_js_on_tab(
            tab_ref,
            js_code,
            purpose="clear highlights",
            executor=executor,  # Pass along the executor if it exists
        )
        # No need for explicit try/except here, helper handles common ones

    async def rehighlight(self, tab_ref: Optional[TabReference]):
        if not tab_ref:
            return

        if self._highlights_active and self._last_highlight_selector and self._last_highlight_color:
            selector = self._last_highlight_selector
            current_color = self._last_highlight_color  # Use the stored color
            tab_id = tab_ref.id
            ws_url = tab_ref.ws_url
            if not ws_url:
                logger.warning(f"Cannot rehighlight on tab {tab_id}: Missing websocket URL.")
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
            # Use helper method for JS execution
            await self._execute_js_on_tab(
                tab_ref, js_code, purpose=f"rehighlight selector '{selector[:30]}...'"
            )
            # Error logging is handled within _execute_js_on_tab
        else:
            pass
            # logger.debug("Skipping rehighlight (not active or no selector/color)")

        # parser rehighlight logic (always independent of _highlights_active)
        if self._parser_last_selector and tab_ref:
            try:
                await self.highlight_parser(
                    tab_ref,
                    self._parser_last_selector,
                    color=self._parser_last_color or "cyan",
                )
            except Exception as e:
                logger.debug(f"Failed to rehighlight parser overlays: {e}")

    def is_active(self) -> bool:
        """Returns true if highlights are considered active."""
        return self._highlights_active

    def set_active(self, active: bool):
        """Explicitly set the active state (e.g., when cancelling agent)."""
        self._highlights_active = active
        if not active:
            self._last_highlight_color = None
            self._last_highlight_selector = None

    async def _execute_js_on_tab(
        self,
        tab_ref: Optional[TabReference],
        js_code: str,
        purpose: str = "generic JS execution",
        executor: Optional[CdpBrowserExecutor] = None,
        timeout: float = 5.0,
    ) -> Optional[Any]:
        """Helper to execute JS, handling common boilerplate and errors."""
        if not tab_ref or not tab_ref.ws_url:
            logger.debug(
                f"Cannot execute JS ({purpose}): Missing active tab reference or websocket URL."
            )
            return None

        tab_id = tab_ref.id
        ws_url = tab_ref.ws_url
        url = tab_ref.url

        temp_executor = None
        if not executor:
            # Create temporary executor if none provided
            # Ensure it's closed or managed properly if CdpBrowserExecutor needs it
            try:
                temp_executor = CdpBrowserExecutor(ws_url, url or "")
                exec_to_use = temp_executor
            except Exception as e:
                logger.error(f"Failed to create temporary executor for JS ({purpose}): {e}")
                return None
        else:
            exec_to_use = executor

        try:
            result = await exec_to_use.evaluate(js_code)
            return result
        except websockets.exceptions.WebSocketException as e:
            logger.warning(
                f"JS execution failed ({purpose}) for tab {tab_id}: WebSocket error - {e}"
            )
            # If using a temp executor, it might be left dangling here. Assuming evaluate handles closure on error.
            return None
        except Exception as e:
            logger.error(
                f"JS execution failed ({purpose}) for tab {tab_id}: Unexpected error - {e}",
                exc_info=True,
            )
            return None
        # Ensure temporary executor is handled if created, though evaluate usually closes it.
        # If CdpBrowserExecutor requires manual close/cleanup, that needs to be added.
        # Assuming evaluate handles connection lifecycle implicitly for now.

    async def show_agent_status(
        self,
        tab_ref: Optional[TabReference],
        status_text: str,
        state: str = "idle",
        show_spinner: bool = False,
        executor: Optional[CdpBrowserExecutor] = None,
    ) -> None:
        """Shows or updates an agent status badge in the top-right corner.

        Optionally includes a simple text spinner.
        """
        if not tab_ref:
            return  # Silently ignore if no tab

        badge_id = self._agent_status_badge_id
        # Define colors based on state
        state_colors = {
            "idle": ("#DDDDDD", "#000000"),  # Light gray background, black text
            "thinking": ("#ADD8E6", "#000000"),  # Light blue background, black text
            "sending": ("#FFFFE0", "#000000"),  # Light yellow background, black text
            "received_success": ("#90EE90", "#000000"),  # Light green background, black text
            "received_no_results": (
                "#FFD700",
                "#000000",
            ),  # Gold/Orange background, black text for no results found
            "received_error": ("#FFA07A", "#000000"),  # Light salmon background, black text
            "final_success": ("#90EE90", "#000000"),  # Same as received_success for now
        }
        bg_color, text_color = state_colors.get(state, state_colors["idle"])

        # Escape status text for JS
        escaped_status_text = (
            status_text.replace("\\\\", "\\\\\\\\")
            .replace("'", "\\\\'")
            .replace('"', '\\\\"')
            .replace("`", "\\\\`")
            .replace(
                "\\\\n", "\\\\\\\\n"
            )  # Ensure newlines are doubly escaped for JS within template literal
        )

        js_code = f"""
        (function() {{
            const badgeId = '{badge_id}';
            const text = `{escaped_status_text}`; // Base text content
            const bgColor = '{bg_color}';
            const textColor = '{text_color}';
            const showSpinner = {str(show_spinner).lower()}; // Correctly pass boolean to JS
            const spinnerChars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']; // Braille spinner
            let spinnerIndex = 0;
            const spinnerIntervalAttr = 'data-spinner-interval-id';
            const baseTextAttr = 'data-base-text';

            let badge = document.getElementById(badgeId);

            // --- Create Badge if it doesn't exist ---
            if (!badge) {{
                badge = document.createElement('div');
                badge.id = badgeId;
                badge.style.position = 'fixed';
                badge.style.top = '0px';
                badge.style.right = '0px';
                badge.style.padding = '2px 5px';
                badge.style.borderRadius = '5px';
                badge.style.fontSize = '10px';
                badge.style.fontFamily = 'sans-serif';
                badge.style.zIndex = '2147483647';
                badge.style.pointerEvents = 'none';
                badge.style.opacity = '0.88';
                badge.style.whiteSpace = 'pre-wrap';
                badge.style.maxWidth = '300px';
                badge.style.border = '1px solid black'; 
                const parent = document.body || document.documentElement;
                if (parent) {{
                    parent.appendChild(badge);
                }} else {{
                    console.warn('Selectron: Could not find parent for badge.');
                    return 'ERROR: Could not find parent for badge.';
                }}
            }}

            // --- Clear existing spinner interval if present ---
            const existingIntervalId = badge.getAttribute(spinnerIntervalAttr);
            if (existingIntervalId) {{
                clearInterval(parseInt(existingIntervalId, 10));
                badge.removeAttribute(spinnerIntervalAttr);
                // Restore base text if spinner was active
                const baseText = badge.getAttribute(baseTextAttr);
                if (baseText) badge.textContent = baseText;
            }}
            badge.removeAttribute(baseTextAttr); // Clear base text attr


            // --- Update style and base text ---
            badge.style.border = '1px solid black'; // Ensure border is set on updates too
            badge.style.backgroundColor = bgColor;
            badge.style.color = textColor;
            badge.textContent = text; // Set initial text

            // --- Start new spinner if requested ---
            if (showSpinner) {{
                badge.setAttribute(baseTextAttr, text); // Store base text
                const intervalId = setInterval(() => {{
                    spinnerIndex = (spinnerIndex + 1) % spinnerChars.length;
                    // Check if badge still exists before updating
                    const currentBadge = document.getElementById(badgeId);
                    if (currentBadge) {{
                       currentBadge.textContent = spinnerChars[spinnerIndex] + ' ' + text;
                    }} else {{
                        // Badge was removed, clear interval
                        clearInterval(intervalId);
                    }}
                }}, 250); // Update spinner every 250ms
                badge.setAttribute(spinnerIntervalAttr, intervalId.toString());
            }}

            return `Agent status badge updated: ${{text}} (State: {state}, Spinner: ${{showSpinner}})`;
        }})();
        """
        await self._execute_js_on_tab(tab_ref, js_code, "update agent status", executor)

    async def hide_agent_status(
        self,
        tab_ref: Optional[TabReference],
        executor: Optional[CdpBrowserExecutor] = None,
    ) -> None:
        """Removes the agent status badge and clears any running spinner."""
        if not tab_ref:
            return  # Silently ignore if no tab

        badge_id = self._agent_status_badge_id
        js_code = f"""
        (function() {{
            const badgeId = '{badge_id}';
            const badge = document.getElementById(badgeId);
            if (badge) {{
                 // --- Clear spinner interval before removing ---
                const spinnerIntervalAttr = 'data-spinner-interval-id';
                const existingIntervalId = badge.getAttribute(spinnerIntervalAttr);
                if (existingIntervalId) {{
                    try {{
                       clearInterval(parseInt(existingIntervalId, 10));
                       console.log('Selectron: Cleared spinner interval on hide.');
                    }} catch (e) {{
                       console.warn('Selectron: Failed to clear spinner interval on hide:', e);
                    }}
                }}
                // --- End clear spinner ---
                try {{
                    badge.remove();
                    return `SUCCESS: Removed agent status badge ('${{badgeId}}').`;
                }} catch (e) {{
                    console.error('Selectron: Failed to remove agent status badge:', e);
                    return `ERROR: Failed to remove agent status badge ('${{badgeId}}'): ${{e.message}}`;
                }}
            }} else {{
                return 'INFO: Agent status badge not found, nothing to remove.';
            }}
        }})();
        """
        await self._execute_js_on_tab(tab_ref, js_code, "hide agent status", executor)

    async def highlight_parser(
        self, tab_ref: Optional[TabReference], selector: str, color: str = "cyan"
    ) -> bool:
        """Highlights elements for a parser definition. These overlays are persistent and
        should *not* be cleared by the normal agent iteration highlights (they use a different
        container). Call :py:meth:`clear_parser` to remove them – typically on URL navigation.

        Args:
            tab_ref: Active tab reference.
            selector: CSS selector to highlight.
            color: Border/background colour (defaults to cyan).
        Returns:
            bool: True if highlight JS executed and returned an expected success string.
        """
        if not tab_ref or not tab_ref.ws_url:
            logger.debug("Cannot highlight parser selector – missing tab reference or ws_url.")
            return False

        # Escape selector for JS template literal
        escaped_selector = (
            selector.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("`", "\\`")
        )

        # dashed border to distinguish
        border_style = f"2px dashed {color}"
        background_color = f"{color}22"  # light transparent fill
        overlay_attr = "data-selectron-parser-overlay"
        container_id = self._parser_container_id

        js_code = f"""
        (function() {{
            const selector = `{escaped_selector}`;
            const borderStyle = '{border_style}';
            const bgColor = '{background_color}';
            const containerId = '{container_id}';
            const overlayAttr = '{overlay_attr}';

            // Ensure container exists
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
                container.style.zIndex = '2147483646'; // just beneath main highlight overlays
                container.style.backgroundColor = 'transparent';
                (document.body || document.documentElement).appendChild(container);
            }} else {{
                // Clear any existing parser overlays first (avoid duplicates)
                const oldOverlays = container.querySelectorAll(`[${{overlayAttr}}="true"]`);
                oldOverlays.forEach(o => o.remove());
            }}

            const elements = document.querySelectorAll(selector);
            if (!elements || elements.length === 0) {{
                return `ParserHighlight: No elements found for selector: ${{selector}}`;
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
                        overlay.style.zIndex = '2147483646';
                        container.appendChild(overlay);
                    }}
                    highlightedCount++;
                }} catch (e) {{
                    console.warn('Selectron parser highlight error:', e);
                }}
            }});
            return `ParserHighlight: Highlighted ${{highlightedCount}} element(s) for: ${{selector}}`;
        }})();
        """

        result = await self._execute_js_on_tab(tab_ref, js_code, purpose="parser highlight")
        if result and isinstance(result, str) and "ParserHighlight" in result:
            # store selector & color for rehighlighting on scroll
            self._parser_last_selector = selector
            self._parser_last_color = color
            return True
        logger.debug(f"Parser highlight JS returned unexpected value: {result}")
        return False

    async def clear_parser(
        self, tab_ref: Optional[TabReference], executor: Optional[CdpBrowserExecutor] = None
    ) -> None:
        """Clears overlays created by :py:meth:`highlight_parser`."""
        if not tab_ref:
            return

        container_id = self._parser_container_id
        js_code = f"""
        (function() {{
            const containerId = '{container_id}';
            const container = document.getElementById(containerId);
            if (container) {{
                try {{
                    container.remove();
                    return `ParserHighlight: removed container '${{containerId}}'.`;
                }} catch (e) {{
                    return `ParserHighlight: ERROR removing container '${{containerId}}' – ${{e.message}}`;
                }}
            }} else {{
                return `ParserHighlight: container '${{containerId}}' not found.`;
            }}
        }})();
        """
        await self._execute_js_on_tab(
            tab_ref, js_code, purpose="clear parser highlight", executor=executor
        )

        # reset parser tracking
        self._parser_last_selector = None
        self._parser_last_color = None

    async def get_elements_html(
        self,
        tab_ref: Optional[TabReference],
        selector: str,
        max_elements: int = 100,
        executor: Optional[CdpBrowserExecutor] = None,
    ) -> Optional[list[str]]:
        """Executes JS in the tab to find elements and return their outerHTML."""
        if not tab_ref or not tab_ref.ws_url:
            logger.debug("Cannot get elements HTML – missing tab reference or ws_url.")
            return None

        escaped_selector = (
            selector.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("`", "\\`")
        )

        js_code = f"""
        (function() {{
            const selector = `{escaped_selector}`;
            const maxCount = {max_elements};
            const elements = document.querySelectorAll(selector);
            const htmlList = [];
            for (let i = 0; i < Math.min(elements.length, maxCount); i++) {{
                htmlList.push(elements[i].outerHTML);
            }}
            // Return as a JSON string to handle potential complexities in HTML content
            return JSON.stringify(htmlList);
        }})();
        """

        result = await self._execute_js_on_tab(
            tab_ref,
            js_code,
            purpose="get elements html",
            executor=executor,
            timeout=10.0,  # Give a bit more time for potentially larger data
        )

        if result and isinstance(result, str):
            try:
                # Parse the JSON string back into a list
                html_list = json.loads(result)
                if isinstance(html_list, list):
                    return html_list
                else:
                    logger.warning(
                        f"JS execution for get_elements_html returned non-list JSON: {result[:100]}..."
                    )
                    return None
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode JSON response from get_elements_html: {e}. Response: {result[:200]}..."
                )
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error processing result from get_elements_html: {e}", exc_info=True
                )
                return None
        else:
            logger.warning(
                f"JS execution for get_elements_html returned unexpected or no result: {type(result)}"
            )
            return None

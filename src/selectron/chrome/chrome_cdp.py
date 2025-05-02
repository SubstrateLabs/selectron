import asyncio
import base64
import binascii
import json
import re
from io import BytesIO
from typing import List, Optional

import httpx
import websockets
from PIL import Image

from selectron.chrome.types import ChromeTab
from selectron.lib.logger import get_logger

logger = get_logger(__name__)

REMOTE_DEBUG_PORT = 9222
_next_message_id = 1


async def get_cdp_websocket_url(port: int = 9222) -> Optional[str]:
    """Fetch the WebSocket debugger URL from Chrome's JSON version endpoint."""
    global _next_message_id
    url = f"http://localhost:{port}/json/version"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=2.0)
            response.raise_for_status()  # Raise an exception for bad status codes
            version_info = response.json()
            ws_url = version_info.get("webSocketDebuggerUrl")
            if ws_url:
                logger.debug(f"Found CDP WebSocket URL: {ws_url}")
                return ws_url
            else:
                logger.warning("Could not find 'webSocketDebuggerUrl' in /json/version response.")
                return None
    except httpx.RequestError as e:
        logger.warning(f"Could not connect to Chrome's debug port at {url}. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching Chrome CDP version info: {e}")
        return None


async def _send_cdp_command(
    ws,
    method: str,
    params: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> Optional[dict]:
    """Send a command to the CDP WebSocket and wait for the specific response."""
    global _next_message_id
    current_id = _next_message_id
    _next_message_id += 1

    command = {"id": current_id, "method": method, "params": params or {}}
    if session_id:
        command["sessionId"] = session_id

    logger.debug(f"Sending CDP command (id={current_id}): {method} {params or ''}")
    await ws.send(json.dumps(command))

    try:
        # Wait for the specific response matching the id
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=30.0)
            response = json.loads(message)
            # logger.debug(f"Received CDP message: {response}") # Too verbose usually
            if response.get("id") == current_id:
                if "error" in response:
                    logger.error(f"CDP command error (id={current_id}): {response['error']}")
                    return None
                logger.debug(f"Received response for id={current_id}")
                return response.get("result")
            elif "method" in response:
                # Handle events or other messages if necessary in the future
                logger.debug(f"Ignoring event/message: {response.get('method')}")
                pass
            else:
                logger.warning(f"Received unexpected message format: {response}")

    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for response to command id {current_id} ({method})")
        return None
    except websockets.exceptions.ConnectionClosed:
        logger.error("WebSocket connection closed unexpectedly.")
        return None
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        return None


async def get_tabs() -> List[ChromeTab]:
    """
    Get all Chrome browser tabs via CDP HTTP API.

    Connects to Chrome DevTools Protocol to retrieve tab information.
    Only returns actual page tabs (not DevTools, extensions, etc).

    Returns:
        List of ChromeTab objects representing open browser tabs
    """
    tabs = []

    try:
        url_to_check = f"http://localhost:{REMOTE_DEBUG_PORT}/json/list"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url_to_check,  # Use variable
                timeout=2.0,  # Use float timeout for httpx
            )
            response.raise_for_status()  # Check for non-2xx status codes
            cdp_tabs_json = response.json()

        # Process each tab
        for tab_info in cdp_tabs_json:
            # Only include actual tabs (type: page), not devtools, etc.
            if tab_info.get("type") == "page":
                # Create a dict with all fields we want to extract
                tab_data = {
                    "id": tab_info.get("id"),
                    "title": tab_info.get("title", "Untitled"),
                    "url": tab_info.get("url", "about:blank"),
                    "webSocketDebuggerUrl": tab_info.get("webSocketDebuggerUrl"),
                    "devtoolsFrontendUrl": tab_info.get("devtoolsFrontendUrl"),
                }

                # Get window ID from debug URL if available
                devtools_url = tab_info.get("devtoolsFrontendUrl", "")
                if "windowId" in devtools_url:
                    try:
                        window_id_match = re.search(r"windowId=(\d+)", devtools_url)
                        if window_id_match:
                            tab_data["window_id"] = int(window_id_match.group(1))
                    except Exception as e:
                        logger.debug(f"Could not extract window ID: {e}")

                # Create Pydantic model instance
                try:
                    tabs.append(ChromeTab(**tab_data))
                except Exception as e:
                    logger.error(f"Failed to parse tab data: {e}")

        return tabs

    # Update exception handling for httpx
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to get tabs: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Chrome DevTools API: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Chrome DevTools API response: {e}")
    except Exception as e:
        logger.error(f"Error getting tabs via Chrome DevTools API: {e}")

    # Return empty list if we couldn't get tabs
    return []


async def get_active_tab_html() -> Optional[str]:
    """Connects to Chrome, finds the first active tab, and retrieves its HTML."""
    ws_url = await get_cdp_websocket_url()
    if not ws_url:
        return None

    try:
        async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
            # 1. Get targets (pages/tabs)
            targets_result = await _send_cdp_command(ws, "Target.getTargets")
            if not targets_result or "targetInfos" not in targets_result:
                logger.error("Failed to get targets from Chrome.")
                return None

            page_targets = [
                t
                for t in targets_result["targetInfos"]
                if t.get("type") == "page" and not t.get("url").startswith("devtools://")
            ]
            if not page_targets:
                logger.warning("No active page targets found.")
                return None

            # Choose the first non-devtools page target
            target_id = page_targets[0]["targetId"]
            target_url = page_targets[0]["url"]
            logger.info(f"Found active page target: ID={target_id}, URL={target_url}")

            # 2. Attach to the target
            attach_result = await _send_cdp_command(
                ws, "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            )
            if not attach_result or "sessionId" not in attach_result:
                logger.error(f"Failed to attach to target {target_id}.")
                return None
            session_id = attach_result["sessionId"]
            logger.debug(f"Attached to target {target_id} with session ID: {session_id}")

            # 3. Execute script to get outerHTML
            script = "document.documentElement.outerHTML"
            eval_result = await _send_cdp_command(
                ws, "Runtime.evaluate", {"expression": script}, session_id=session_id
            )

            # Detach is important, do it even if eval fails
            detach_result = await _send_cdp_command(
                ws, "Target.detachFromTarget", {"sessionId": session_id}
            )
            if (
                detach_result is None
            ):  # Checks for explicit None, indicating an error during send/recv
                logger.warning(
                    f"Failed to properly detach from session {session_id}. Might be okay."
                )

            if not eval_result or "result" not in eval_result:
                logger.error(f"Failed to evaluate script in target {target_id}.")
                return None

            if eval_result["result"].get("type") == "string":
                html_content = eval_result["result"].get("value")
                logger.info(f"Successfully retrieved HTML content (length: {len(html_content)}).")
                return html_content
            else:
                logger.error(
                    f"Script evaluation did not return a string: {eval_result['result'].get('type')} / {eval_result['result'].get('description')}"
                )
                return None

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {ws_url}")
        return None
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_active_tab_html: {e}", exc_info=True)
        return None


async def capture_active_tab_screenshot(
    output_dir: str = ".",
    filename: Optional[str] = None,
    format: str = "png",
    quality: Optional[int] = None,
) -> Optional[Image.Image]:
    """Connects to Chrome, finds the first active tab, and captures a screenshot.

    Args:
        output_dir: Directory to save the screenshot.
        filename: Base name for the screenshot file (timestamp added if None).
        format: Image format (png, jpeg, webp). Defaults to png.
        quality: Compression quality (0-100) for jpeg. Defaults to None.

    Returns:
        A PIL Image object, or None if failed.
    """
    ws_url = await get_cdp_websocket_url()
    if not ws_url:
        return None

    if format not in ["png", "jpeg", "webp"]:
        logger.error(f"Invalid screenshot format: {format}. Use png, jpeg, or webp.")
        return None

    try:
        async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
            # --- Reuse logic to find and attach to target --- #
            targets_result = await _send_cdp_command(ws, "Target.getTargets")
            if not targets_result or "targetInfos" not in targets_result:
                logger.error("Failed to get targets from Chrome.")
                return None

            page_targets = [
                t
                for t in targets_result["targetInfos"]
                if t.get("type") == "page" and not t.get("url").startswith("devtools://")
            ]
            if not page_targets:
                logger.warning("No active page targets found for screenshot.")
                return None

            target_id = page_targets[0]["targetId"]
            target_url = page_targets[0]["url"]
            logger.info(
                f"Found active page target for screenshot: ID={target_id}, URL={target_url}"
            )

            attach_result = await _send_cdp_command(
                ws, "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            )
            if not attach_result or "sessionId" not in attach_result:
                logger.error(f"Failed to attach to target {target_id} for screenshot.")
                return None
            session_id = attach_result["sessionId"]
            logger.debug(f"Attached to target {target_id} with session ID: {session_id}")
            # --- End reuse --- #

            # 4. Capture Screenshot
            screenshot_params: dict[str, str | int] = {"format": format}
            if format == "jpeg" and quality is not None:
                screenshot_params["quality"] = max(0, min(100, quality))

            capture_result = await _send_cdp_command(
                ws, "Page.captureScreenshot", screenshot_params, session_id=session_id
            )

            # Detach is important, do it even if capture fails
            detach_result = await _send_cdp_command(
                ws, "Target.detachFromTarget", {"sessionId": session_id}
            )
            if (
                detach_result is None
            ):  # Checks for explicit None, indicating an error during send/recv
                logger.warning(
                    f"Failed to properly detach from session {session_id} after screenshot attempt. Might be okay."
                )

            if not capture_result or "data" not in capture_result:
                logger.error(f"Failed to capture screenshot in target {target_id}.")
                return None

            # 5. Decode and Save
            image_data_base64 = capture_result["data"]
            try:
                image_data = base64.b64decode(image_data_base64)
            except (TypeError, binascii.Error) as e:
                logger.error(f"Failed to decode base64 image data: {e}")
                return None

            # Convert bytes to PIL Image
            try:
                image = Image.open(BytesIO(image_data))
                logger.info(
                    f"Successfully captured screenshot and created PIL Image (format: {format}, size: {image.size})."
                )
                return image
            except Exception as e:
                logger.error(f"Failed to create PIL Image from screenshot data: {e}")
                return None

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {ws_url}")
        return None
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket connection error during screenshot: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in capture_active_tab_screenshot: {e}",
            exc_info=True,
        )
        return None


# Example usage (optional, for direct testing)
if __name__ == "__main__":

    async def run_test():
        print("Attempting to get active tab HTML...")
        html = await get_active_tab_html()
        if html:
            print("\nSuccessfully retrieved HTML:")
            print("=" * 30)
            print(html[:500] + "..." if len(html) > 500 else html)  # Print first 500 chars
            print("=" * 30)
        else:
            print("\nFailed to retrieve HTML.")

    asyncio.run(run_test())

import asyncio
import base64
import binascii
import json
import re
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
import websockets
from PIL import Image

from selectron.chrome.types import ChromeTab
from selectron.util.logger import get_logger

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


async def send_cdp_command(
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

    await ws.send(json.dumps(command))

    try:
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=20.0)
            response = json.loads(message)
            if response.get("id") == current_id:
                if "error" in response:
                    logger.error(f"CDP command error (id={current_id}): {response['error']}")
                    return None
                return response.get("result")
    except asyncio.TimeoutError:
        # logger.error(f"Timeout waiting for response to command id {current_id} ({method})")
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
            targets_result = await send_cdp_command(ws, "Target.getTargets")
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
            attach_result = await send_cdp_command(
                ws, "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            )
            if not attach_result or "sessionId" not in attach_result:
                logger.error(f"Failed to attach to target {target_id}.")
                return None
            session_id = attach_result["sessionId"]
            logger.debug(f"Attached to target {target_id} with session ID: {session_id}")

            # 3. Execute script to get outerHTML
            script = "document.documentElement.outerHTML"
            eval_result = await send_cdp_command(
                ws, "Runtime.evaluate", {"expression": script}, session_id=session_id
            )

            # Detach is important, do it even if eval fails
            detach_result = await send_cdp_command(
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


async def wait_for_page_load(ws, timeout: float = 15.0) -> bool:
    """Checks readyState and waits for Page.loadEventFired if necessary."""
    try:
        # 1. Check current readyState first for quick exit
        eval_result = await send_cdp_command(
            ws, "Runtime.evaluate", {"expression": "document.readyState", "returnByValue": True}
        )
        # Use returnByValue: True to ensure we get the string value directly
        if eval_result is None:
            logger.warning("Failed to get document.readyState.")
            # Decide how to proceed: treat as not loaded or return False?
            # Let's assume we should try waiting for the load event anyway.
            pass  # Fall through to waiting logic
        elif eval_result.get("result", {}).get("value") == "complete":
            return True

        logger.debug(
            f"Page readyState is '{eval_result.get('result', {}).get('value') if eval_result else 'unknown'}', waiting for Page.loadEventFired..."
        )

        # 2. Enable Page domain events
        enable_result = await send_cdp_command(ws, "Page.enable")
        if enable_result is None:  # Check if enable command itself failed
            logger.error("Failed to enable Page domain events.")
            return False

        # 3. Wait for the load event
        start_time = asyncio.get_event_loop().time()
        while True:
            # Check for overall timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"Timeout waiting for Page.loadEventFired after {timeout}s.")
                await send_cdp_command(ws, "Page.disable")  # Attempt disable
                return False  # Indicate timeout/failure

            try:
                # Wait for *any* message, with a short timeout to allow checking the overall loop timeout
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                response = json.loads(message)

                # Is it the event we are waiting for?
                if response.get("method") == "Page.loadEventFired":
                    logger.debug("Received Page.loadEventFired event.")
                    await send_cdp_command(ws, "Page.disable")  # Disable after success
                    return True  # Success

                # Ignore other messages (command responses, other events)
                # logger.debug(f"Ignoring message while waiting for load: {response.get('method') or response.get('id')}")

            except asyncio.TimeoutError:
                # Timeout on ws.recv() is normal, just continue loop to check overall timeout
                continue
            except json.JSONDecodeError:
                logger.warning("Failed to decode message while waiting for load event.")
                continue  # Try receiving next message
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket closed while waiting for load event.")
                # No need to disable Page domain, connection is gone
                return False  # Connection lost
            except Exception as inner_e:
                logger.error(f"Error processing message while waiting for load: {inner_e}")
                # Attempt disable on other errors
                await send_cdp_command(ws, "Page.disable")
                return False

    except Exception as e:
        logger.error(f"Error in _wait_for_page_load: {e}", exc_info=True)
        # Attempt to disable Page domain in case of outer error, ignore failure
        try:
            await send_cdp_command(ws, "Page.disable")
        except Exception:
            pass
        return False


async def get_tab_html(ws_url: str, settle_delay_s: float = 0.0) -> Optional[str]:
    """Connects to a specific tab's debugger WebSocket URL and retrieves its HTML,
    waiting for the page load event first.
    """
    if not ws_url:
        logger.error("get_tab_html called with empty ws_url.")
        return None

    try:
        # Connect directly to the tab's debugger URL
        async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
            # Wait for page load event
            loaded = await wait_for_page_load(ws)
            if not loaded:
                logger.warning(
                    f"Page load event not detected/timed out for {ws_url}, proceeding to get HTML anyway..."
                )
                # Decide whether to return None or proceed. Let's proceed.

            # Optional settle delay after load
            if settle_delay_s > 0:
                logger.debug(f"Waiting for settle delay: {settle_delay_s}s after load wait.")
                await asyncio.sleep(settle_delay_s)

            # Execute script to get outerHTML - no need to attach/detach with direct connection
            script = "document.documentElement.outerHTML"
            eval_result = await send_cdp_command(ws, "Runtime.evaluate", {"expression": script})

            if not eval_result or "result" not in eval_result:
                logger.error(f"Failed to evaluate script in tab with ws_url: {ws_url}.")
                return None

            if eval_result["result"].get("type") == "string":
                html_content = eval_result["result"].get("value")
                return html_content
            else:
                logger.error(
                    f"Script evaluation did not return a string: {eval_result['result'].get('type')} / {eval_result['result'].get('description')} for {ws_url}"
                )
                return None

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {ws_url}")
        return None
    except websockets.exceptions.WebSocketException as e:
        # Log specific connection errors like refused, timeout etc.
        logger.error(f"WebSocket connection error to {ws_url}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_tab_html for {ws_url}: {e}", exc_info=True
        )
        return None


async def get_final_url_and_title(
    ws: Any, initial_url: str, initial_title: str, tab_id_for_logging: Optional[str] = None
) -> Tuple[str, str]:
    """Gets the final URL and title of the page after load and redirects using an existing websocket.

    Args:
        ws: The active WebSocket connection to the tab.
        initial_url: The URL known before potential redirects.
        initial_title: The title known before potential changes.
        tab_id_for_logging: Optional tab ID for more informative logging.

    Returns:
        A tuple containing the (final_url, final_title).
    """
    final_url = initial_url
    final_title = initial_title
    log_prefix = f"tab {tab_id_for_logging}" if tab_id_for_logging else "tab"

    # Get Final URL
    try:
        url_script = "window.location.href"
        url_eval = await send_cdp_command(
            ws, "Runtime.evaluate", {"expression": url_script, "returnByValue": True}
        )
        if url_eval and url_eval.get("result", {}).get("type") == "string":
            final_url = url_eval["result"]["value"]
            if final_url != initial_url:
                logger.info(
                    f"URL changed after load for {log_prefix}: {initial_url} -> {final_url}"
                )
            else:
                logger.debug(f"URL confirmed after load for {log_prefix}: {final_url}")
        else:
            logger.warning(
                f"Could not get final URL for {log_prefix}. Using initial: {initial_url}"
            )
            final_url = initial_url  # Fallback
    except Exception as url_e:
        logger.error(f"Error getting final URL for {log_prefix}: {url_e}", exc_info=True)
        final_url = initial_url  # Fallback

    # Get Final Title
    try:
        title_script = "document.title"
        title_eval = await send_cdp_command(
            ws, "Runtime.evaluate", {"expression": title_script, "returnByValue": True}
        )
        if title_eval and title_eval.get("result", {}).get("type") == "string":
            final_title = title_eval["result"]["value"]
        else:
            logger.warning(
                f"Could not get final title for {log_prefix}. Using initial: {initial_title}"
            )
            final_title = initial_title  # Fallback
    except Exception as title_e:
        logger.warning(
            f"Error getting final title for {log_prefix}: {title_e}. Using initial: {initial_title}"
        )
        final_title = initial_title  # Fallback

    return final_url, final_title


async def get_html_via_ws(ws: Any, url_for_logging: str) -> Optional[str]:
    """Fetches the HTML content of the page using an existing WebSocket connection.

    Args:
        ws: The active WebSocket connection to the tab.
        url_for_logging: The URL to use in log messages.

    Returns:
        The HTML string, or None if an error occurred.
    """
    try:
        html_script = "document.documentElement.outerHTML"
        html_eval = await send_cdp_command(ws, "Runtime.evaluate", {"expression": html_script})
        if html_eval and html_eval.get("result", {}).get("type") == "string":
            html = html_eval["result"].get("value")
            if html:
                logger.debug(
                    f"Retrieved HTML via ws for URL {url_for_logging} (Length: {len(html)})"
                )
            else:
                # Should not happen if type is string, but safety check
                logger.warning(
                    f"Retrieved HTML via ws for {url_for_logging}, but content is unexpectedly None/empty."
                )
            return html
        else:
            logger.warning(
                f"Could not retrieve HTML via ws for {url_for_logging}. Eval result: {html_eval}"
            )
            return None
    except Exception as html_e:
        logger.error(
            f"Error getting HTML via WebSocket for {url_for_logging}: {html_e}", exc_info=True
        )
        return None


async def capture_tab_screenshot(
    ws_url: str,
    format: str = "png",
    quality: Optional[int] = None,
    settle_delay_s: float = 0.0,
    ws_connection: Optional[Any] = None,
) -> Optional[Image.Image]:
    """Connects to a specific tab's debugger WebSocket URL (or uses provided connection)
    and captures a screenshot, waiting for the page load event first.

    Args:
        ws_url: The WebSocket debugger URL (used only if ws_connection is None).
        format: Image format (png, jpeg, webp). Defaults to png.
        quality: Compression quality (0-100) for jpeg/webp. Defaults to None.
        settle_delay_s: Delay in seconds after waiting for the page load event.
        ws_connection: An optional existing WebSocket connection to use.

    Returns:
        A PIL Image object, or None if failed.
    """
    if not ws_url and not ws_connection:
        logger.error("capture_tab_screenshot called with neither ws_url nor ws_connection.")
        return None
    if ws_connection and ws_connection.state == websockets.protocol.State.CLOSED:
        logger.error("capture_tab_screenshot called with a closed ws_connection.")
        return None

    # Internal function to handle the core logic using a connection
    async def _do_capture(ws: Any) -> Optional[Image.Image]:
        try:
            # Wait for page load event (only makes sense if we didn't just connect)
            # If ws_connection is passed, assume caller handles load state? Maybe skip wait?
            # For now, let's keep the wait, but it might be redundant if handler waited.
            loaded = await wait_for_page_load(ws)
            if not loaded:
                logger.warning(
                    "Page load event not detected/timed out, proceeding to screenshot anyway..."
                )

            if settle_delay_s > 0:
                logger.debug(f"Waiting for settle delay: {settle_delay_s}s after load wait.")
                await asyncio.sleep(settle_delay_s)

            screenshot_params: dict[str, str | int] = {"format": format}
            if format == "jpeg" or format == "webp":
                screenshot_params["quality"] = quality if quality is not None else 90

            capture_result = await send_cdp_command(ws, "Page.captureScreenshot", screenshot_params)

            if not capture_result or "data" not in capture_result:
                logger.error("Failed to capture screenshot via connection.")
                return None

            image_data_base64 = capture_result["data"]
            try:
                image_data = base64.b64decode(image_data_base64)
            except (TypeError, binascii.Error) as e:
                logger.error(f"Failed to decode base64 image data: {e}")
                return None

            try:
                image = Image.open(BytesIO(image_data))
                return image
            except Exception as e:
                logger.error(f"Failed to create PIL Image from screenshot data: {e}")
                return None
        except Exception as e:
            logger.error(f"Error during screenshot capture logic: {e}", exc_info=True)
            return None

    # Use provided connection or create a new one
    try:
        if ws_connection:
            # Directly use the provided connection without context manager
            return await _do_capture(ws_connection)
        else:
            # Establish a new connection using async with
            logger.debug("Establishing new WebSocket connection for screenshot.")
            if not ws_url:
                logger.error("ws_url is required when ws_connection is not provided.")
                return None  # Should be caught earlier, but safety check
            async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws_new:
                return await _do_capture(ws_new)

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {ws_url}")
        return None
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket connection error during screenshot for {ws_url}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in capture_tab_screenshot setup/connection for {ws_url}: {e}",
            exc_info=True,
        )
        return None


async def capture_active_tab_screenshot(
    output_dir: str = ".",
    filename: Optional[str] = None,
    format: str = "png",
    quality: Optional[int] = None,
    settle_delay_s: float = 0.0,
) -> Optional[Image.Image]:
    """Connects to Chrome, finds the first active tab, and captures a screenshot.

    Args:
        output_dir: Directory to save the screenshot.
        filename: Base name for the screenshot file (timestamp added if None).
        format: Image format (png, jpeg, webp). Defaults to png.
        quality: Compression quality (0-100) for jpeg. Defaults to None.
        settle_delay_s: Delay in seconds after waiting for the page load event.

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
            targets_result = await send_cdp_command(ws, "Target.getTargets")
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

            attach_result = await send_cdp_command(
                ws, "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            )
            if not attach_result or "sessionId" not in attach_result:
                logger.error(f"Failed to attach to target {target_id} for screenshot.")
                return None
            session_id = attach_result["sessionId"]
            logger.debug(f"Attached to target {target_id} with session ID: {session_id}")
            # --- End reuse --- #

            # 4. Capture Screenshot
            loaded = await wait_for_page_load(ws)
            if not loaded:
                logger.warning(
                    f"Page load event not detected/timed out for active tab {target_id}, proceeding anyway..."
                )

            if settle_delay_s > 0:
                logger.debug(f"Waiting for settle delay: {settle_delay_s}s after load wait.")
                await asyncio.sleep(settle_delay_s)

            screenshot_params: dict[str, str | int] = {"format": format}
            if format == "jpeg" and quality is not None:
                screenshot_params["quality"] = max(0, min(100, quality))

            capture_result = await send_cdp_command(
                ws, "Page.captureScreenshot", screenshot_params, session_id=session_id
            )

            # Detach is important, do it even if capture fails
            detach_result = await send_cdp_command(
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


async def monitor_user_interactions(ws_url: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Monitor clicks and scrolls in a tab using CDP and yield events.

    Connects to the tab's WebSocket, injects JS listeners, and listens
    for console messages indicating user interaction.

    Yields:
        dict: Structured event data like {"type": "click", "data": {...}} or {"type": "scroll", "data": {...}}
    """
    connection = None  # Keep track of the connection to close it reliably
    ws = None  # Initialize ws outside the try block
    try:
        max_retries = 3
        retry_delay = 0.5  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                connection = await websockets.connect(
                    ws_url,
                    open_timeout=5.0,  # Timeout for each attempt
                    close_timeout=5.0,
                    max_size=20 * 1024 * 1024,
                )
                ws = connection
                break
            except (
                OSError,
                websockets.exceptions.InvalidMessage,
                asyncio.TimeoutError,
            ) as conn_err:
                logger.warning(
                    f"ws connection attempt {attempt + 1}/{max_retries} failed for {ws_url}: {type(conn_err).__name__}",
                    exc_info=False,
                )
                if attempt + 1 == max_retries:
                    logger.error(
                        f"Max retries reached for WebSocket connection to {ws_url}. Giving up."
                    )
                    return  # Exit generator if all retries fail
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        # If loop finishes without connecting (shouldn't happen due to return above, but safety check)
        if not ws:
            logger.error(f"Failed to establish WebSocket connection after retries for {ws_url}.")
            return

        # === Step 1: Enable Runtime domain FIRST ===
        await send_cdp_command(ws, "Runtime.enable")
        # === Step 2: Enable dependent domains (Page, Log) ===
        await send_cdp_command(ws, "Page.enable")
        # === Step 3: Subscribe to consoleAPICalled event ===
        # Inject JS listeners
        js_code = """
        (function() {
            console.log('BROCC_DEBUG: Injecting listeners...'); // Debug log
            // Use a closure to prevent polluting the global scope too much
            let lastScrollTimestamp = 0;
            let lastClickTimestamp = 0;
            const DEBOUNCE_MS = 250; // Only log if events are spaced out

            document.addEventListener('click', e => {
                const now = Date.now();
                if (now - lastClickTimestamp > DEBOUNCE_MS) {
                    const clickData = {
                        x: e.clientX,
                        y: e.clientY,
                        target: e.target ? e.target.tagName : 'unknown',
                        timestamp: now
                    };
                    console.log('BROCC_CLICK_EVENT', JSON.stringify(clickData));
                    lastClickTimestamp = now;
                }
            }, { capture: true, passive: true }); // Use capture phase, non-blocking

            document.addEventListener('scroll', e => {
                 const now = Date.now();
                 if (now - lastScrollTimestamp > DEBOUNCE_MS) {
                    const scrollData = {
                        scrollX: window.scrollX,
                        scrollY: window.scrollY,
                        timestamp: now
                    };
                    console.log('BROCC_SCROLL_EVENT', JSON.stringify(scrollData));
                    lastScrollTimestamp = now;
                 }
            }, { capture: true, passive: true }); // Use capture phase, non-blocking

            console.log('BROCC_DEBUG: Listeners successfully installed.'); // Debug log
            return "Interaction listeners installed.";
        })();
        """
        _eval_result = await send_cdp_command(
            ws,
            "Runtime.evaluate",
            {"expression": js_code, "awaitPromise": False, "returnByValue": True},
        )
        # Listen for console entries
        while True:
            response_raw = await ws.recv()
            response = json.loads(response_raw)

            if response.get("method") == "Runtime.consoleAPICalled":
                call_type = response.get("params", {}).get("type")
                args = response.get("params", {}).get("args", [])

                # Check if it's a log message with our specific prefix
                if call_type == "log" and len(args) >= 1:
                    first_arg_value = args[0].get("value")

                    # --- Handle BROCC_DEBUG messages ---
                    if first_arg_value == "BROCC_DEBUG: Injecting listeners...":
                        pass
                        # logger.info(f"[{ws_url[-10:]}] JS Injection: Starting setup.")
                    elif first_arg_value == "BROCC_DEBUG: Listeners successfully installed.":
                        pass
                        # logger.info(
                        #     f"[{ws_url[-10:]}] JS Injection: Listeners confirmed installed."
                        # )
                    # --- Handle BROCC_CLICK_EVENT ---
                    elif first_arg_value == "BROCC_CLICK_EVENT" and len(args) >= 2:
                        try:
                            click_data = json.loads(args[1].get("value", "{}"))
                            # Ensure scrollY is included, default to 0 if not (though unlikely for click)
                            click_data.setdefault("scrollY", 0)
                            yield {"type": "click", "data": click_data}
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse click event data from CDP console")
                    # --- Handle BROCC_SCROLL_EVENT ---
                    elif first_arg_value == "BROCC_SCROLL_EVENT" and len(args) >= 2:
                        try:
                            scroll_data = json.loads(args[1].get("value", "{}"))
                            raw_scroll_y = scroll_data.get("scrollY")
                            # Check if scrollY is a number (int or float)
                            if isinstance(raw_scroll_y, (int, float)):
                                # Convert to int and update the dict before yielding
                                scroll_data["scrollY"] = int(raw_scroll_y)
                                yield {"type": "scroll", "data": scroll_data}
                            else:
                                logger.warning(
                                    f"Received invalid scroll data format (scrollY not a number): {scroll_data}"
                                )
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse scroll event data from CDP console")

    except (
        websockets.ConnectionClosedOK,
        websockets.ConnectionClosedError,
        websockets.ConnectionClosed,
    ) as e:
        # These are expected closures, log as info or debug
        logger.info(f"ws connection closed for {ws_url}: {e}")
    # Keep generic exception for unexpected errors during the loop
    except Exception as e:
        # Log errors happening *after* successful connection as ERROR
        logger.error(
            f"Error during interaction monitoring for {ws_url}: {type(e).__name__} - {e}",
            exc_info=True,
        )
    finally:
        # Check state before attempting to close
        if connection and connection.state != websockets.protocol.State.CLOSED:
            await connection.close()
        # This generator stops yielding when an error occurs or connection closes.

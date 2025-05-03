import asyncio
import base64
import binascii
import json
from io import BytesIO
from typing import Any, Optional

import websockets
from PIL import Image
from websockets.protocol import State

from selectron.chrome.chrome_cdp import send_cdp_command
from selectron.dom.dom_service import BrowserExecutor
from selectron.util.logger import get_logger

logger = get_logger(__name__)


class CdpBrowserExecutor(BrowserExecutor):
    """Implements BrowserExecutor using Chrome DevTools Protocol.

    Can either manage its own connection based on ws_url or use an externally provided one.
    """

    def __init__(self, ws_url: str, tab_url: str, ws_connection: Optional[Any] = None):
        self.ws_url = ws_url  # Still needed for logging/identification
        self._tab_url = tab_url
        self._provided_ws: Optional[Any] = ws_connection  # Revert to Any
        self._internal_ws: Optional[Any] = None  # Revert to Any
        self._lock = asyncio.Lock()  # To manage internal connection state

    @property
    def _ws(self) -> Optional[Any]:  # Revert to Any
        """Returns the active WebSocket connection (provided or internal)."""
        return self._provided_ws if self._provided_ws is not None else self._internal_ws

    async def _connect(self):
        """Establishes internal WebSocket connection if not already connected and none was provided."""
        # Only connect if no external connection is provided
        if self._provided_ws is not None:
            return

        async with self._lock:
            if self._internal_ws is None or self._internal_ws.closed:
                try:
                    self._internal_ws = await websockets.connect(
                        self.ws_url, max_size=30 * 1024 * 1024, open_timeout=10, close_timeout=10
                    )
                    # Enable Runtime domain immediately after internal connection
                    # Use self._internal_ws directly here as self._ws might return the (None) provided_ws
                    await send_cdp_command(self._internal_ws, "Runtime.enable")
                except (
                    websockets.exceptions.WebSocketException,
                    OSError,
                    asyncio.TimeoutError,
                ) as e:
                    logger.error(
                        f"(Internal Connect) Failed to connect to CDP WebSocket {self.ws_url}: {e}"
                    )
                    self._internal_ws = None
                    raise

    async def _disconnect(self):
        """Closes the internal WebSocket connection if it was established."""
        # Only disconnect if no external connection was provided
        if self._provided_ws is not None:
            return

        async with self._lock:
            if self._internal_ws and not self._internal_ws.closed:
                logger.debug(
                    f"(Internal Disconnect) Disconnecting from CDP WebSocket: {self.ws_url}"
                )
                await self._internal_ws.close()
                self._internal_ws = None
            else:
                logger.debug(
                    "(Internal Disconnect) WebSocket already closed or not internally connected."
                )

    async def _send_command(self, method: str, params: Optional[dict] = None) -> Optional[dict]:
        """Ensures connection (if managed internally) and sends a CDP command."""
        active_ws = self._ws  # Get the currently active connection (provided or internal)
        if active_ws is None:
            # If no connection provided, attempt internal connection
            if self._provided_ws is None:
                await self._connect()
                active_ws = self._internal_ws  # Re-check after connect attempt
            else:
                # Connection was provided but is None/closed - this is an issue
                logger.error("Provided WebSocket connection is not valid.")
                return None

        # Check connection state using the State enum
        if not active_ws or active_ws.state == State.CLOSED:
            logger.error("Cannot send command, WebSocket is not connected or closed.")
            return None

        try:
            # Send command using the determined active connection
            return await send_cdp_command(active_ws, method, params)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(
                f"WebSocket connection closed while sending command {method}. Error: {e}"
            )
            # If the connection was internal, mark it as closed
            if active_ws == self._internal_ws:
                async with self._lock:
                    self._internal_ws = None
            # Do not attempt reconnect here; let the caller handle it if necessary.
            # If it was a provided connection, the provider needs to handle it.
            raise

    async def evaluate(self, expression: str, arg: Optional[dict] = None) -> Any:
        """Evaluates JavaScript expression in the page context."""
        # Ensure Runtime is enabled (send_command handles connection)
        # Send Runtime.enable just in case, it's idempotent
        await self._send_command("Runtime.enable")

        if arg is not None:
            # Fallback or default: Runtime.evaluate (less ideal for complex args)
            params = {"expression": expression, "awaitPromise": True, "returnByValue": True}
            # Simple serialization for basic args - beware of complex objects
            expression_with_arg = f"({expression})(JSON.parse('{json.dumps(arg)}'))"
            params["expression"] = expression_with_arg
            eval_result = await self._send_command("Runtime.evaluate", params)
        else:
            # No arguments, use simple evaluate
            params = {"expression": expression, "awaitPromise": True, "returnByValue": True}
            eval_result = await self._send_command("Runtime.evaluate", params)
        # Process result (common logic for both cases)
        if eval_result and "result" in eval_result:
            result_data = eval_result["result"]
            if result_data.get("type") != "undefined":
                return result_data.get("value")
            else:
                logger.debug(
                    f"[evaluate] JS result type is undefined for expr starting with '{expression[:100]}...'. Raw result: {eval_result}. Returning None."
                )  # Log if undefined
                return None  # Represent undefined JS result as None
        elif eval_result and "exceptionDetails" in eval_result:
            exception_details = eval_result["exceptionDetails"]
            logger.error(f"JavaScript exception during evaluate: {exception_details}")
            raise RuntimeError(
                f"JavaScript exception: {exception_details.get('text', 'Unknown JS Error')}"
            )
        else:
            return None

    async def capture_screenshot(
        self, format: str = "png", quality: Optional[int] = None
    ) -> Optional[Image.Image]:
        """Captures a screenshot using the current page context and connection."""
        active_ws = self._ws
        if not active_ws or active_ws.state == State.CLOSED:
            logger.error("Cannot capture screenshot, WebSocket is not connected or closed.")
            return None
        try:
            # Ensure Page domain is enabled (might be redundant, but safe)
            # await self._send_command("Page.enable") # Enable might interfere if called elsewhere? Let's try without first.
            screenshot_params: dict[str, Any] = {"format": format}
            if (format == "jpeg" or format == "webp") and quality is not None:
                screenshot_params["quality"] = max(0, min(100, quality))

            # Use the existing _send_command which handles connection checks
            capture_result = await self._send_command("Page.captureScreenshot", screenshot_params)

            if not capture_result or "data" not in capture_result:
                logger.error("Failed to capture screenshot via CDP command.")
                return None

            image_data_base64 = capture_result["data"]
            try:
                image_data = base64.b64decode(image_data_base64)
                image = Image.open(BytesIO(image_data))
                logger.debug(f"Screenshot captured successfully (format: {format})")
                return image
            except (TypeError, binascii.Error) as decode_err:
                logger.error(f"Failed to decode base64 image data: {decode_err}")
                return None
            except Exception as img_err:
                logger.error(f"Failed to create PIL Image from screenshot data: {img_err}")
                return None

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket closed during screenshot attempt: {e}")
            # If internal, mark as closed
            if active_ws == self._internal_ws:
                async with self._lock:
                    self._internal_ws = None
            raise  # Re-raise so caller knows connection died
        except Exception as e:
            logger.error(f"Unexpected error during screenshot capture: {e}", exc_info=True)
            return None

    @property
    def url(self) -> str:
        """Gets the URL of the current page (potentially stale)."""
        return self._tab_url

    # Context manager support only makes sense for internally managed connections
    async def __aenter__(self):
        if self._provided_ws is not None:
            logger.warning(
                "CdpBrowserExecutor.__aenter__ called with a provided WebSocket connection. Connection will not be managed."
            )
            return self  # Return self, but don't connect
        await self._connect()  # Connect internal ws
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._provided_ws is not None:
            return  # Do nothing if connection was provided
        await self._disconnect()  # Disconnect internal ws

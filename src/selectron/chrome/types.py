from typing import NamedTuple, Optional

from pydantic import BaseModel, Field


class TabReference(NamedTuple):
    """Minimal reference to identify a tab and its relevant state."""

    id: str
    url: str
    html: Optional[str] = None
    title: Optional[str] = None
    ws_url: Optional[str] = None


class ChromeTab(BaseModel):
    id: str
    title: str = Field(default="Untitled")
    url: str = Field(default="about:blank")
    window_id: Optional[int] = None
    webSocketDebuggerUrl: Optional[str] = None
    devtoolsFrontendUrl: Optional[str] = None

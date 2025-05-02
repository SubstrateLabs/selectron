from selectron.chrome.diff_tabs import diff_tabs
from selectron.chrome.types import ChromeTab, TabReference


# Helper function to create dummy TabReference
def create_ref(id: str, url: str, title: str = "Title") -> TabReference:
    return TabReference(id=id, url=url, title=title, html=None)


# Helper function to create dummy ChromeTab
def create_tab(
    id: str, url: str, title: str = "Title", ws_url: str | None = "ws://debug"
) -> ChromeTab:
    return ChromeTab(id=id, title=title, url=url, webSocketDebuggerUrl=ws_url, window_id=1)


# --- Test Cases ---


def test_no_changes():
    prev_refs = {"t1": create_ref("t1", "url1"), "t2": create_ref("t2", "url2")}
    curr_tabs = {"t1": create_tab("t1", "url1"), "t2": create_tab("t2", "url2")}
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert added == []
    assert removed == []
    assert navigated == []


def test_only_added():
    prev_refs = {"t1": create_ref("t1", "url1")}
    curr_tabs = {
        "t1": create_tab("t1", "url1"),
        "t2": create_tab("t2", "url2"),
        "t3": create_tab("t3", "url3"),
    }
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert len(added) == 2
    assert added[0].id == "t2" or added[1].id == "t2"  # Order not guaranteed
    assert added[0].id == "t3" or added[1].id == "t3"
    assert removed == []
    assert navigated == []


def test_only_removed():
    prev_refs = {
        "t1": create_ref("t1", "url1"),
        "t2": create_ref("t2", "url2"),
        "t3": create_ref("t3", "url3"),
    }
    curr_tabs = {"t1": create_tab("t1", "url1")}
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert added == []
    assert len(removed) == 2
    assert removed[0].id == "t2" or removed[1].id == "t2"
    assert removed[0].id == "t3" or removed[1].id == "t3"
    assert navigated == []


def test_only_navigated():
    prev_refs = {
        "t1": create_ref("t1", "url1_old"),
        "t2": create_ref("t2", "url2_old"),
    }
    curr_tabs = {"t1": create_tab("t1", "url1_new"), "t2": create_tab("t2", "url2_new")}
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert added == []
    assert removed == []
    assert len(navigated) == 2
    # Check content of navigated pairs (order doesn't matter)
    nav_t1 = next(pair for pair in navigated if pair[0].id == "t1")
    nav_t2 = next(pair for pair in navigated if pair[0].id == "t2")
    assert nav_t1[0].url == "url1_new"
    assert nav_t1[1].url == "url1_old"
    assert nav_t2[0].url == "url2_new"
    assert nav_t2[1].url == "url2_old"


def test_mixed_changes():
    prev_refs = {
        "t1": create_ref("t1", "url1"),  # Keep
        "t2": create_ref("t2", "url2_old"),  # Navigate
        "t3": create_ref("t3", "url3"),  # Remove
    }
    curr_tabs = {
        "t1": create_tab("t1", "url1"),  # Keep
        "t2": create_tab("t2", "url2_new"),  # Navigate
        "t4": create_tab("t4", "url4"),  # Add
    }
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)

    assert len(added) == 1
    assert added[0].id == "t4"
    assert added[0].url == "url4"

    assert len(removed) == 1
    assert removed[0].id == "t3"
    assert removed[0].url == "url3"

    assert len(navigated) == 1
    assert navigated[0][0].id == "t2"
    assert navigated[0][0].url == "url2_new"
    assert navigated[0][1].url == "url2_old"


def test_empty_previous():
    prev_refs = {}
    curr_tabs = {"t1": create_tab("t1", "url1"), "t2": create_tab("t2", "url2")}
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert len(added) == 2
    assert removed == []
    assert navigated == []


def test_empty_current():
    prev_refs = {"t1": create_ref("t1", "url1"), "t2": create_ref("t2", "url2")}
    curr_tabs = {}
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert added == []
    assert len(removed) == 2
    assert navigated == []


def test_both_empty():
    prev_refs = {}
    curr_tabs = {}
    added, removed, navigated = diff_tabs(curr_tabs, prev_refs)
    assert added == []
    assert removed == []
    assert navigated == []

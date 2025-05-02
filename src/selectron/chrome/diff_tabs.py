from typing import Dict, List, Tuple

from selectron.chrome.types import ChromeTab, TabReference


def diff_tabs(
    current_tabs_map: Dict[str, ChromeTab],
    previous_refs_map: Dict[str, TabReference],
) -> Tuple[List[ChromeTab], List[TabReference], List[Tuple[ChromeTab, TabReference]]]:
    """Compares current tabs to previous references to find changes."""
    current_ids = set(current_tabs_map.keys())
    previous_ids = set(previous_refs_map.keys())

    added_tab_ids = current_ids - previous_ids
    removed_tab_ids = previous_ids - current_ids
    potentially_navigated_ids = previous_ids.intersection(current_ids)

    added_tabs = [current_tabs_map[tab_id] for tab_id in added_tab_ids]
    removed_refs = [previous_refs_map[tab_id] for tab_id in removed_tab_ids]
    navigated_pairs = []

    for tab_id in potentially_navigated_ids:
        current_tab = current_tabs_map[tab_id]
        previous_ref = previous_refs_map[tab_id]
        if current_tab.url != previous_ref.url:
            navigated_pairs.append((current_tab, previous_ref))

    return added_tabs, removed_refs, navigated_pairs

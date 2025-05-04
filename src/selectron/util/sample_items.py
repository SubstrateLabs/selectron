import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from selectron.util.logger import get_logger

logger = get_logger(__name__)


def sample_items(
    items: List[Dict[str, Any]], sample_size: int = 4
) -> List[Tuple[int, Dict[str, Any]]]:
    """Samples a small, structurally diverse set of dicts from a list.

    Prioritizes sampling one item of each distinct key shape (frozenset of keys)
    before randomly sampling remaining items up to `sample_size`.
    Useful for getting a varied preview of data structures.
    Skips items that are not JSON-serializable during shape calculation.
    """
    if not items or sample_size <= 0:
        return []

    sampled_item_tuples: List[Tuple[int, Dict[str, Any]]] = []
    items_by_shape: Dict[frozenset[str], List[Dict[str, Any]]] = defaultdict(list)

    # Group items by their key structure (shape)
    serializable_items_indices: List[int] = []
    for idx, item_dict in enumerate(items):
        try:
            # Use JSON dumps with sort_keys for a canonical shape representation
            # This also implicitly checks serializability
            _ = json.dumps(item_dict, sort_keys=True)
            shape = frozenset(item_dict.keys())
            items_by_shape[shape].append(item_dict)
            serializable_items_indices.append(idx)
        except TypeError:
            logger.debug(f"skipping non-serializable item at index {idx} during sampling")
            continue

    if not items_by_shape:
        logger.warning("no json-serializable items found for sampling")
        return []

    # Prioritize one sample from each distinct shape
    distinct_shapes = list(items_by_shape.keys())
    random.shuffle(distinct_shapes)
    sampled_indices: set[int] = set()
    for shape in distinct_shapes:
        if len(sampled_item_tuples) >= sample_size:
            break
        chosen_item = random.choice(items_by_shape[shape])
        # Find the first original index of this chosen item
        try:
            original_index = items.index(chosen_item)
            sampled_item_tuples.append((original_index, chosen_item))
            sampled_indices.add(original_index)  # Add index to sampled set
        except ValueError:
            # Should not happen if chosen_item came from items_by_shape
            logger.warning(f"chosen item not found in original list: {chosen_item}")

        # Avoid re-sampling the exact same item later if possible
        # by removing it from the pool if multiple items had the same shape
        items_by_shape[shape].remove(chosen_item)
        if not items_by_shape[shape]:
            del items_by_shape[shape]  # remove shape if list is empty

    # If still below sample_size, add more random items by index
    if len(sampled_item_tuples) < sample_size:
        # Find indices of serializable items that haven't been sampled yet
        available_indices = [i for i in serializable_items_indices if i not in sampled_indices]
        random.shuffle(available_indices)

        needed = sample_size - len(sampled_item_tuples)
        for idx_to_add in available_indices[:needed]:
            # Add the item by its index, allowing identical values
            sampled_item_tuples.append((idx_to_add, items[idx_to_add]))

    return sampled_item_tuples

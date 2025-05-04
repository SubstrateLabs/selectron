from typing import Any, Dict, List, Set

from bs4 import BeautifulSoup


def _flatten(val: Any) -> List[str]:
    """Helper to flatten nested values (lists, dicts) into a list of strings."""
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple, set)):
        out: List[str] = []
        for it in val:
            out.extend(_flatten(it))
        return out
    if isinstance(val, dict):
        out: List[str] = []
        for sub in val.values():
            out.extend(_flatten(sub))
        return out
    return []


def validate_empty_columns(outputs: List[Dict[str, Any]], keys: Set[str]) -> List[str]:
    """Check for keys that only ever have empty values across all outputs.
    Returns a list of feedback strings for problematic keys.
    """
    feedback: List[str] = []
    for key in keys:
        is_always_empty = True
        found_key_at_least_once = False
        for output_dict in outputs:
            if key in output_dict:
                found_key_at_least_once = True
                value = output_dict[key]
                # Define "non-empty": not None, not "", not [], not {}
                if value not in (None, "", [], {}):
                    is_always_empty = False
                    break
        if found_key_at_least_once and is_always_empty:
            feedback.append(
                f"Key '{key}' exists but has only empty values (e.g., '', [], {{}}, None) across all results. Consider removing it or fixing the extraction."
            )
    return feedback


def validate_identical_columns(outputs: List[Dict[str, Any]], keys: Set[str]) -> List[str]:
    """Check for keys that have the same non-empty value across all outputs where they appear.

    Returns a list of feedback strings for problematic keys.
    """
    feedback: List[str] = []
    for key in keys:
        first_value: Any = None
        found_key_more_than_once = False
        is_always_identical = True
        initial_value_set = False
        count = 0

        for output_dict in outputs:
            if key in output_dict:
                count += 1
                if not initial_value_set:
                    first_value = output_dict[key]
                    initial_value_set = True

        if count > 1:
            found_key_more_than_once = True
            for output_dict in outputs:
                if key in output_dict:
                    if output_dict[key] != first_value:
                        is_always_identical = False
                        break

        if found_key_more_than_once and is_always_identical:
            if first_value not in (None, "", [], {}):
                feedback.append(
                    f"Key '{key}' has the identical non-empty value '{repr(first_value)[:50]}...' across all {count} results where it appears. Is this intended?"
                )
    return feedback


def validate_text_representation(
    outputs: List[Dict[str, Any]], html_samples: List[str]
) -> List[str]:
    """Ensure each output has at least one value fuzzily matching the element's visible text.
    Returns a list of feedback strings for problematic samples.
    """
    feedback: List[str] = []
    samples_without_text_match: List[int] = []
    for idx, output_dict in enumerate(outputs):
        parsing_succeeded = False  # Track if text extraction worked
        plain_text = ""  # Initialize plain_text
        try:
            # Need try-except as bs4 can sometimes fail on fragments
            soup = BeautifulSoup(html_samples[idx], "html.parser")
            # Heuristic: If soup seems minimal/fragmentary, treat as parsing failure for this validation
            if not soup.find(True):  # Check if *any* tag was parsed
                raise ValueError("Parsed soup seems empty or is just text.")
            plain_text = soup.get_text(" ", strip=True).lower()
            if not plain_text:  # Also treat empty text result as skippable
                raise ValueError("Parsed text is empty.")
            parsing_succeeded = True  # Mark success only if soup and text seem valid
        except Exception:  # Catch broader exceptions including our ValueError
            # If text extraction or sanity check fails, skip validation for this sample
            continue

        # --- This check is now less necessary due to explicit continues --- #
        # if not parsing_succeeded or not plain_text:
        #      continue
        # ----------------------------------------------------------------- #

        plain_tokens = set(plain_text.split())
        has_match = False

        for val in output_dict.values():
            for s in _flatten(val):
                s_low = s.lower().strip()
                if not s_low:
                    continue
                # Direct substring
                if s_low in plain_text or plain_text in s_low:
                    has_match = True
                    break
                # Token overlap
                s_tokens = set(s_low.split())
                if s_tokens and len(s_tokens & plain_tokens) / len(s_tokens) >= 0.5:
                    has_match = True
                    break
            if has_match:
                break
        # Only flag if parsing succeeded AND no match was found
        if parsing_succeeded and not has_match:
            samples_without_text_match.append(idx)

    if samples_without_text_match:
        feedback.append(
            "Outputs for samples "
            + ", ".join(map(str, samples_without_text_match))
            + " lack any value approximating the element's visible text. Consider adding a text-rich field (e.g., title/description)."
        )
    return feedback


def validate_redundant_key_pairs(outputs: List[Dict[str, Any]], keys: Set[str]) -> List[str]:
    """Check pairs of keys for consistent redundancy across all outputs.
    Returns a list of feedback strings for problematic key pairs.
    """
    feedback: List[str] = []
    # No point checking redundancy with 0 or 1 output dicts
    if len(outputs) <= 1:
        return feedback

    checked_pairs: Set[frozenset[str]] = set()
    for key1 in keys:
        for key2 in keys:
            if key1 == key2 or frozenset([key1, key2]) in checked_pairs:
                continue

            is_redundant = True
            found_pair = False
            for output_dict in outputs:
                if key1 in output_dict and key2 in output_dict:
                    found_pair = True
                    # Compare values (handle various types implicitly with !=)
                    if output_dict[key1] != output_dict[key2]:
                        is_redundant = False
                        break

            if found_pair and is_redundant:
                feedback.append(
                    f"Keys '{key1}' and '{key2}' appear to have identical values across all results where both are present. Consider merging or removing one."
                )
            checked_pairs.add(frozenset([key1, key2]))
    return feedback


def validate_cross_key_duplicates(outputs: List[Dict[str, Any]], keys: Set[str]) -> List[str]:
    """Check for redundancy between list keys and singular keys.

    Specifically checks:
        - if `primary_url` value exists in the `urls` list.
        - if `author_avatar_url` value exists as a `src` in the `images` list.

    Returns a list of feedback strings for detected redundancies.
    """
    feedback: List[str] = []
    primary_url_in_urls = False
    avatar_in_images = False

    for output_dict in outputs:
        # Check primary_url vs urls
        primary = output_dict.get("primary_url")
        urls = output_dict.get("urls")
        if primary and isinstance(urls, list) and primary in urls:
            primary_url_in_urls = True

        # Check author_avatar_url vs images
        avatar_url = output_dict.get("author_avatar_url")
        images = output_dict.get("images")
        if avatar_url and isinstance(images, list):
            for img_dict in images:
                if isinstance(img_dict, dict) and img_dict.get("src") == avatar_url:
                    avatar_in_images = True
                    break  # Found redundancy for this output, check next output

        # Early exit if both found
        if primary_url_in_urls and avatar_in_images:
            break

    if primary_url_in_urls:
        feedback.append(
            "Redundancy detected: The value of `primary_url` was also found within the `urls` list. Ensure `urls` contains only *other* links."
        )
    if avatar_in_images:
        feedback.append(
            "Redundancy detected: The value of `author_avatar_url` was also found as a `src` within the `images` list. Ensure `images` excludes the author avatar."
        )

    return feedback

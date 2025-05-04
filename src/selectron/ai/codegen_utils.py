from typing import Any, Dict, List, Set, Tuple

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

        plain_tokens = set(plain_text.split())
        has_match = False

        for val in output_dict.values():
            # Flatten value in case it's a list/dict containing strings
            for s in _flatten(val):
                s_low = str(s).lower().strip()  # Ensure string and normalize
                if not s_low:
                    continue

                min_match_len = 25
                overlap_threshold = 0.6

                # Only proceed if the extracted string is reasonably long
                if len(s_low) < min_match_len:
                    continue

                # Check 1: Direct Substring
                if s_low in plain_text or plain_text in s_low:
                    has_match = True
                    break  # Found a match for this output dict

                # Check 2: Token Overlap
                s_tokens = set(s_low.split())
                # Ensure s_tokens is not empty before division
                if (
                    plain_tokens
                    and s_tokens
                    and (len(s_tokens & plain_tokens) / len(s_tokens)) >= overlap_threshold
                ):
                    has_match = True
                    break  # Found a match for this output dict

            if has_match:
                break  # Exit outer loop once a match is found for the sample

        if parsing_succeeded and not has_match:
            samples_without_text_match.append(idx)

    if samples_without_text_match:
        feedback.append(
            "Outputs for samples "
            + ", ".join(map(str, samples_without_text_match))
            + " did not contain any single string value that is both reasonably long (>=25 chars) and sufficiently similar (>=60% token overlap or substring) to the element's overall visible text. Ensure the main content is extracted cleanly into an appropriate field."
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


def _check_word_repetition(text: str, sequence_len: int = 7, min_words: int = 14) -> bool:
    """Check if a string contains a repeated sequence of `sequence_len` words.

    Args:
        text: The input string.
        sequence_len: The number of consecutive words to check for repetition.
        min_words: The minimum number of words the text must have to be checked.

    Returns:
        True if a repeated sequence is found, False otherwise.
    """
    words = text.lower().split()
    if len(words) < min_words:
        return False  # Not enough words for meaningful repetition check

    sequences = set()
    for i in range(len(words) - sequence_len + 1):
        seq = tuple(words[i : i + sequence_len])
        if seq in sequences:
            return True  # Found a repeated sequence
        sequences.add(seq)

    return False


def validate_internal_repetition(outputs: List[Dict[str, Any]], keys: Set[str]) -> List[str]:
    """Check string values for significant internal word sequence repetition.

    Returns a list of feedback strings for keys with detected repetition.
    """
    feedback: List[str] = []
    repetitive_keys: Set[str] = set()

    for output_dict in outputs:
        for key in keys:
            if key in repetitive_keys:
                continue  # Already flagged this key

            value = output_dict.get(key)
            if isinstance(value, str):
                if _check_word_repetition(value):
                    repetitive_keys.add(key)
                    # No need to check other outputs for this key once repetition is found
                    # break # Optional: break inner loop if perf is critical

    for key in repetitive_keys:
        feedback.append(
            f"Key '{key}' appears to contain significant internal repetition (sequence of words repeated) in at least one result. Check extraction logic."
        )

    return feedback


def validate_naive_text_match(outputs: List[Dict[str, Any]], html_samples: List[str]) -> List[str]:
    """Check if any extracted string value is identical to the naive full text extraction.

    This helps detect cases where the agent might have just used `get_text()`
    on a large container instead of extracting specific content.

    Returns a list of feedback strings for keys matching the naive text.
    """
    feedback: List[str] = []
    naive_match_keys: Set[str] = set()

    for idx, output_dict in enumerate(outputs):
        try:
            soup = BeautifulSoup(html_samples[idx], "html.parser")
            # Ensure soup parsing didn't completely fail
            if not soup.find(True):
                continue
            naive_text = soup.get_text(separator=" ", strip=True)
            if not naive_text:  # Skip if the element genuinely has no text
                continue
        except Exception:
            continue  # Skip if HTML parsing fails

        for key, value in output_dict.items():
            if key in naive_match_keys:
                continue  # Already flagged this key

            # Only check string values
            if isinstance(value, str):
                # Normalize whitespace before exact match
                normalized_value = " ".join(value.strip().split())
                normalized_naive_text = " ".join(naive_text.split())
                # Compare based on word sequence, ignoring inter-word spacing differences
                if normalized_value and normalized_value == normalized_naive_text:
                    naive_match_keys.add(key)
                    # break # NOTE: Optional optimization?

    for key in naive_match_keys:
        feedback.append(
            f"Key '{key}' in at least one result is identical to the element's full naive text (`get_text()`). Refine extraction to be more specific and exclude metadata/irrelevant text."
        )

    return feedback


def clean_agent_code(agent_output: Any) -> str:
    """Cleans potential code output from an agent.

    Handles cases where the agent might return:
    - Raw Python code string.
    - Code wrapped in markdown fences.
    - A dictionary containing the code string as a value (attempts extraction).
    - Other types (attempts string conversion).

    Returns:
        The cleaned code string.
    """
    code_str = ""
    if isinstance(agent_output, dict):
        # Try to find a value that looks like Python code
        found_code = False
        potential_code = None  # Store potential code here
        for value in agent_output.values():
            # Heuristic: Check for common Python keywords
            if isinstance(value, str) and value.strip():
                if "def " in value or "import " in value:
                    potential_code = value
                    found_code = True
                    break  # Found likely code, stop searching

        if found_code:
            code_str = potential_code
        else:
            # Fallback: No code-like string found, return string representation of the dict
            code_str = str(agent_output)
    else:
        # Assume it's already a string or convertible
        code_str = str(agent_output)

    # Strip markdown fences
    if code_str is None:  # Should not happen with current logic, but safety check
        code_str = ""
    cleaned_code = code_str.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[len("```python") :].strip()
    elif cleaned_code.startswith("```"):
        cleaned_code = cleaned_code[len("```") :].strip()
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[: -len("```")].strip()

    return cleaned_code


def validate_result(obj: Any) -> Tuple[bool, str]:
    """Ensure obj is a non-empty dict with string keys and allowed value types.

    Allowed value types:
    - str
    - int
    - list[Any] (recursively validated so elements are allowed types)
    - dict[str, str]
    """

    def _is_valid_val(val: Any) -> bool:
        if isinstance(val, str):
            return True
        if isinstance(val, int):
            return True
        if isinstance(val, dict):
            # Check for dict[str, str]
            return all(isinstance(k, str) and isinstance(v, str) for k, v in val.items())
        if isinstance(val, list):
            # Explicitly check for list[str | dict[str, str | int | None]]
            for item in val:
                is_item_str = isinstance(item, str)
                # Check if item is a dict where keys are str and values are str, int, or None
                is_item_dict_flexible_values = isinstance(item, dict) and all(
                    isinstance(k, str) and isinstance(v_inner, (str, int)) or v_inner is None
                    for k, v_inner in item.items()
                )
                if not (is_item_str or is_item_dict_flexible_values):
                    return False  # Item is neither str nor dict[str, str | int | None]
            return True  # All items were valid str or dict[str, str | int | None]
        return False

    if not isinstance(obj, dict):
        return False, "result is not a dict"
    if not obj:
        return False, "dict is empty"
    for k, v in obj.items():
        if not isinstance(k, str):
            return False, f"non-string key detected: {k!r}"
        if not _is_valid_val(v):
            # If validation fails and the value is a list, find the invalid item
            if isinstance(v, list):
                for idx, item in enumerate(v):
                    is_item_str = isinstance(item, str)
                    # Check if item is a dict where keys are str and values are str, int, or None
                    is_item_dict_flexible_values = isinstance(item, dict) and all(
                        isinstance(k, str) and isinstance(v_inner, (str, int)) or v_inner is None
                        for k, v_inner in item.items()
                    )
                    if not (is_item_str or is_item_dict_flexible_values):
                        # Construct specific feedback for list item failure
                        return (
                            False,
                            f"invalid item at index {idx} for key '{k}'. Expected str or dict[str, str | int | None], but got {type(item)} with invalid internal types.",
                        )
                # Should not be reached if _is_valid_val returned False, but as a fallback:
                return (
                    False,
                    f"invalid list value for key '{k}' (reason unclear, check list items).",
                )
            else:
                # Generic feedback for non-list types
                if v is None:
                    return (
                        False,
                        f"invalid value for key '{k}': assigned None. Omit the key entirely if no valid value is found.",
                    )
                else:
                    return False, f"invalid value for key '{k}': type {type(v)}"
    return True, "ok"

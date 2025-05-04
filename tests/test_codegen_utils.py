from selectron.ai.codegen_utils import (
    _flatten,
    validate_cross_key_duplicates,
    validate_empty_columns,
    validate_identical_columns,
    validate_redundant_key_pairs,
    validate_text_representation,
)

# --- Tests for _flatten ---


def test_flatten_string():
    assert _flatten("hello") == ["hello"]


def test_flatten_list_of_strings():
    assert _flatten(["a", "b", "c"]) == ["a", "b", "c"]


def test_flatten_dict_of_strings():
    assert _flatten({"k1": "v1", "k2": "v2"}) == ["v1", "v2"]


def test_flatten_nested_list():
    assert _flatten(["a", ["b", "c"], "d"]) == ["a", "b", "c", "d"]


def test_flatten_nested_dict():
    assert _flatten({"k1": "v1", "k2": {"nk1": "nv1"}}) == ["v1", "nv1"]


def test_flatten_mixed_nested():
    assert _flatten(["a", {"k1": ["b", "c"]}, 123, None]) == ["a", "b", "c"]  # non-strings ignored


def test_flatten_empty_list():
    assert _flatten([]) == []


def test_flatten_empty_dict():
    assert _flatten({}) == []


def test_flatten_non_iterable():
    assert _flatten(123) == []
    assert _flatten(None) == []


# --- Tests for validate_empty_columns ---


def test_validate_empty_columns_all_empty():
    outputs = [{"a": "", "b": 1}, {"a": None, "b": 2}]
    keys = {"a", "b"}
    feedback = validate_empty_columns(outputs, keys)
    assert len(feedback) == 1
    assert "'a'" in feedback[0]


def test_validate_empty_columns_partially_empty():
    outputs = [{"a": "", "b": 1}, {"a": "hello", "b": 2}]
    keys = {"a", "b"}
    feedback = validate_empty_columns(outputs, keys)
    assert len(feedback) == 0


def test_validate_empty_columns_never_empty():
    outputs = [{"a": "1", "b": 1}, {"a": "2", "b": 2}]
    keys = {"a", "b"}
    feedback = validate_empty_columns(outputs, keys)
    assert len(feedback) == 0


def test_validate_empty_columns_key_missing_sometimes():
    outputs = [{"a": ""}, {"a": None, "b": 1}]
    keys = {"a", "b"}
    feedback = validate_empty_columns(outputs, keys)
    assert len(feedback) == 1
    assert "'a'" in feedback[0]


def test_validate_empty_columns_no_outputs():
    outputs = []
    keys = {"a", "b"}
    feedback = validate_empty_columns(outputs, keys)
    assert len(feedback) == 0


def test_validate_empty_columns_zero_value():
    outputs = [{"a": 0, "b": 1}]
    keys = {"a", "b"}
    feedback = validate_empty_columns(outputs, keys)
    assert len(feedback) == 0  # 0 is not considered empty


# --- Tests for validate_identical_columns ---


def test_validate_identical_columns_all_identical_non_empty():
    outputs = [{"a": "same", "b": 1}, {"a": "same", "b": 2}]
    keys = {"a", "b"}
    feedback = validate_identical_columns(outputs, keys)
    assert len(feedback) == 1
    assert "'a'" in feedback[0]
    assert "same" in feedback[0]


def test_validate_identical_columns_all_identical_empty():
    outputs = [{"a": "", "b": 1}, {"a": "", "b": 2}]
    keys = {"a", "b"}
    feedback = validate_identical_columns(outputs, keys)
    assert len(feedback) == 0  # Identical empty is handled by empty check


def test_validate_identical_columns_different_values():
    outputs = [{"a": "one", "b": 1}, {"a": "two", "b": 2}]
    keys = {"a", "b"}
    feedback = validate_identical_columns(outputs, keys)
    assert len(feedback) == 0


def test_validate_identical_columns_key_missing_sometimes():
    outputs = [{"a": "same"}, {"a": "same", "b": 1}]
    keys = {"a", "b"}
    feedback = validate_identical_columns(outputs, keys)
    assert len(feedback) == 1
    assert "'a'" in feedback[0]


def test_validate_identical_columns_single_output():
    outputs = [{"a": "same", "b": 1}]
    keys = {"a", "b"}
    feedback = validate_identical_columns(outputs, keys)
    assert len(feedback) == 0  # Only checks if count > 1


def test_validate_identical_columns_no_outputs():
    outputs = []
    keys = {"a", "b"}
    feedback = validate_identical_columns(outputs, keys)
    assert len(feedback) == 0


# --- Tests for validate_text_representation ---


def test_validate_text_rep_match_substring():
    outputs = [{"title": "quick brown", "url": "/fox"}]
    html = ["<div>the quick brown fox jumps</div>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0


def test_validate_text_rep_match_token_overlap():
    outputs = [{"desc": "brown fox lazy dog", "id": "1"}]
    html = ["<p>the quick <b>brown</b> fox jumps over the lazy dog</p>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0  # "brown fox lazy dog" has 4 tokens, all 4 are in html text


def test_validate_text_rep_no_match():
    outputs = [{"id": "123", "url": "/path"}]
    html = ["<div>visible text here</div>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 1
    assert "samples 0" in feedback[0]
    assert "visible text" in feedback[0]


def test_validate_text_rep_multiple_samples_one_fail():
    outputs = [{"title": "good"}, {"id": "bad"}]
    html = ["<span>good text</span>", "<span>other text</span>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 1
    assert "samples 1" in feedback[0]


def test_validate_text_rep_empty_html_text():
    outputs = [{"id": "1"}]
    html = ["<div></div>"]  # No text content
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0  # Skips validation if no text


def test_validate_text_rep_empty_output_value():
    outputs = [{"title": ""}]
    html = ["<div>some text</div>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 1  # Empty string doesn't match


def test_validate_text_rep_match_nested_value():
    outputs = [{"data": {"nested": "fox jumps"}, "id": 1}]
    html = ["<div>the quick brown fox jumps over</div>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0


def test_validate_text_rep_bs4_fail_gracefully():
    outputs = [{"title": "some title"}]
    html = ["<unclosed"]  # Invalid html fragment
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0  # Should skip validation for this sample


# --- Tests for validate_redundant_key_pairs ---


def test_validate_redundant_keys_present():
    outputs = [{"a": "1", "b": "1", "c": "x"}, {"a": "2", "b": "2", "c": "y"}]
    keys = {"a", "b", "c"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 1
    assert ("'a'" in feedback[0] and "'b'" in feedback[0]) or (
        "'b'" in feedback[0] and "'a'" in feedback[0]
    )


def test_validate_redundant_keys_not_redundant():
    outputs = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    keys = {"a", "b"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 0


def test_validate_redundant_keys_partially_present_redundant():
    outputs = [{"a": "1", "b": "1"}, {"a": "2", "b": "2", "c": "y"}]
    keys = {"a", "b", "c"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 1  # a and b are still redundant where they appear together
    assert ("'a'" in feedback[0] and "'b'" in feedback[0]) or (
        "'b'" in feedback[0] and "'a'" in feedback[0]
    )


def test_validate_redundant_keys_partially_present_not_redundant():
    outputs = [{"a": "1", "b": "x"}, {"a": "2", "c": "y"}]
    keys = {"a", "b", "c"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 0  # a and b never appear together


def test_validate_redundant_keys_no_outputs():
    outputs = []
    keys = {"a", "b"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 0


def test_validate_redundant_keys_single_output():
    outputs = [{"a": "1", "b": "1"}]
    keys = {"a", "b"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 0  # Only checks if pair found in multiple outputs implicitly


def test_validate_redundant_keys_different_types_but_equal():
    outputs = [{"a": 1, "b": 1.0}, {"a": 0, "b": 0.0}]
    keys = {"a", "b"}
    feedback = validate_redundant_key_pairs(outputs, keys)
    assert len(feedback) == 1  # 1 == 1.0 and 0 == 0.0
    assert ("'a'" in feedback[0] and "'b'" in feedback[0]) or (
        "'b'" in feedback[0] and "'a'" in feedback[0]
    )


# --- Tests for validate_cross_key_duplicates ---


def test_validate_cross_key_dup_primary_url_in_urls():
    outputs = [
        {"primary_url": "/a", "urls": ["/a", "/b"]},
        {"primary_url": "/c", "urls": ["/d", "/c"]},
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 1
    assert "primary_url` was also found within the `urls` list" in feedback[0]


def test_validate_cross_key_dup_avatar_in_images():
    outputs = [
        {
            "author_avatar_url": "avatar.jpg",
            "images": [{"src": "img1.jpg"}, {"src": "avatar.jpg"}],
        },
        {
            "author_avatar_url": "avatar2.png",
            "images": [{"src": "avatar2.png"}],
        },
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 1
    assert "author_avatar_url` was also found as a `src` within the `images` list" in feedback[0]


def test_validate_cross_key_dup_both():
    outputs = [
        {
            "primary_url": "/a",
            "urls": ["/a", "/b"],
            "author_avatar_url": "avatar.jpg",
            "images": [{"src": "img1.jpg"}, {"src": "avatar.jpg"}],
        }
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 2
    assert "primary_url` was also found within the `urls` list" in feedback[0]
    assert "author_avatar_url` was also found as a `src` within the `images` list" in feedback[1]


def test_validate_cross_key_dup_none():
    outputs = [
        {
            "primary_url": "/a",
            "urls": ["/b", "/c"],
            "author_avatar_url": "avatar.jpg",
            "images": [{"src": "img1.jpg"}, {"src": "img2.jpg"}],
        }
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 0


def test_validate_cross_key_dup_missing_keys():
    outputs = [
        {"primary_url": "/a"},  # Missing urls
        {"urls": ["/b"]},  # Missing primary_url
        {"author_avatar_url": "avatar.jpg"},  # Missing images
        {"images": [{"src": "img1.jpg"}]},  # Missing author_avatar_url
        {
            "primary_url": "/c",
            "urls": ["/d"],
            "author_avatar_url": "avatar2.png",
            "images": [{"src": "img2.png"}],
        },  # No duplication
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 0


def test_validate_cross_key_dup_empty_lists():
    outputs = [
        {"primary_url": "/a", "urls": []},
        {"author_avatar_url": "avatar.jpg", "images": []},
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 0


def test_validate_cross_key_dup_wrong_types():
    outputs = [
        {"primary_url": "/a", "urls": "not_a_list"},
        {"author_avatar_url": "avatar.jpg", "images": "not_a_list"},
        {
            "author_avatar_url": "avatar2.png",
            "images": ["not_a_dict", {"alt": "no_src"}],
        },
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 0  # Should ignore malformed data


def test_validate_cross_key_dup_single_output():
    outputs = [
        {
            "primary_url": "/a",
            "urls": ["/a"],
            "author_avatar_url": "avatar.jpg",
            "images": [{"src": "avatar.jpg"}],
        }
    ]
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 2  # Checks apply even with one output


def test_validate_cross_key_dup_no_outputs():
    outputs = []
    feedback = validate_cross_key_duplicates(outputs, set())
    assert len(feedback) == 0

from selectron.ai.codegen_utils import (
    _check_word_repetition,
    _flatten,
    clean_agent_code,
    validate_cross_key_duplicates,
    validate_empty_columns,
    validate_identical_columns,
    validate_internal_repetition,
    validate_naive_text_match,
    validate_redundant_key_pairs,
    validate_result,
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
    outputs = [{"title": "the quick brown fox jumps over", "url": "/fox"}]
    html = ["<div>the quick brown fox jumps over the lazy dog</div>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0


def test_validate_text_rep_match_token_overlap():
    outputs = [{"desc": "the quick brown fox jumps over the lazy dog, yes it does", "id": "1"}]
    html = ["<p>the quick <b>brown</b> fox jumps over the lazy dog</p>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 0


def test_validate_text_rep_no_match():
    outputs = [{"id": "123", "url": "/path"}]
    html = ["<div>visible text here</div>"]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 1
    assert "samples 0" in feedback[0]
    assert "visible text" in feedback[0]


def test_validate_text_rep_multiple_samples_one_fail():
    outputs = [{"title": "good text should pass this length check"}, {"id": "bad"}]
    html = [
        "<span>good text should pass this length check easily</span>",
        "<span>other text</span>",
    ]
    feedback = validate_text_representation(outputs, html)
    assert len(feedback) == 1


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
    assert len(feedback) == 1


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


# --- Tests for _check_word_repetition ---


def test_check_word_repetition_present():
    text = "this is a test string with a repeated sequence this is a test string with words"
    assert _check_word_repetition(text, sequence_len=5, min_words=10) is True


def test_check_word_repetition_not_present():
    text = "this is a perfectly normal string without any obvious repetition of word sequences"
    assert _check_word_repetition(text, sequence_len=5, min_words=10) is False


def test_check_word_repetition_too_short():
    text = "too short"
    assert _check_word_repetition(text, sequence_len=5, min_words=10) is False


def test_check_word_repetition_exact_min_words_no_rep():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    assert _check_word_repetition(text, sequence_len=7, min_words=14) is False


def test_check_word_repetition_exact_min_words_with_rep():
    text = "one two three four five six seven one two three four five six seven"
    assert _check_word_repetition(text, sequence_len=7, min_words=14) is True


def test_check_word_repetition_case_insensitive():
    text = "Test test TEST test Test TEST test"
    assert _check_word_repetition(text, sequence_len=2, min_words=4) is True


def test_check_word_repetition_empty_string():
    text = ""
    assert _check_word_repetition(text) is False


# --- Tests for validate_internal_repetition ---


def test_validate_internal_rep_present():
    outputs = [
        {
            "desc": "one two three four five six seven makes a sequence one two three four five six seven",
            "other": "ok",
        },
        {"desc": "fine"},
    ]
    keys = {"desc", "other"}
    feedback = validate_internal_repetition(outputs, keys)
    assert len(feedback) == 1
    assert "'desc'" in feedback[0]


def test_validate_internal_rep_not_present():
    outputs = [{"desc": "this text is fine", "title": "so is this"}]
    keys = {"desc", "title"}
    feedback = validate_internal_repetition(outputs, keys)
    assert len(feedback) == 0


def test_validate_internal_rep_non_string():
    outputs = [{"desc": ["list", "items"], "count": 5}]
    keys = {"desc", "count"}
    feedback = validate_internal_repetition(outputs, keys)
    assert len(feedback) == 0


def test_validate_internal_rep_multiple_keys_repetitive():
    outputs = [
        {
            "desc": "rep rep rep rep rep rep rep rep rep rep rep rep rep rep",
            "title": "title title title title title title title title title title title title title title",
        }
    ]
    keys = {"desc", "title"}
    feedback = validate_internal_repetition(outputs, keys)
    assert len(feedback) == 2
    assert any("'desc'" in f for f in feedback)
    assert any("'title'" in f for f in feedback)


def test_validate_internal_rep_no_outputs():
    outputs = []
    keys = {"desc"}
    feedback = validate_internal_repetition(outputs, keys)
    assert len(feedback) == 0


def test_validate_internal_rep_key_missing_sometimes():
    outputs = [
        {"desc": "rep rep rep rep rep rep rep rep rep rep rep rep rep rep"},
        {"title": "ok"},
    ]
    keys = {"desc", "title"}
    feedback = validate_internal_repetition(outputs, keys)
    assert len(feedback) == 1
    assert "'desc'" in feedback[0]


# --- Tests for validate_naive_text_match ---


def test_validate_naive_match_present():
    outputs = [
        {
            "desc": "Author Name @handle · 5h Main content here Button Text 12 34 5K",
            "other": "Specific info",
        }
    ]
    html = [
        "<div><span>Author Name @handle · 5h</span><p>Main content here</p><span>Button Text</span><span>12</span> <span>34</span> <span>5K</span></div>"
    ]
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 1
    assert "'desc'" in feedback[0]


def test_validate_naive_match_not_present():
    outputs = [
        {
            "desc": "Main content here",
            "author": "Author Name @handle",
            "timestamp": "5h",
        }
    ]
    html = ["<div><span>Author Name @handle · 5h</span><p>Main content here</p></div>"]
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 0


def test_validate_naive_match_empty_string_val():
    outputs = [{"desc": ""}]
    html = ["<div>Some text</div>"]
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 0  # Empty string won't match non-empty naive text


def test_validate_naive_match_empty_html():
    outputs = [{"desc": "something"}]
    html = ["<div></div>"]
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 0  # Naive text is empty, won't match


def test_validate_naive_match_parsing_fail():
    outputs = [{"desc": "something"}]
    html = ["<unclosed>"]
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 0  # Should skip validation


def test_validate_naive_match_multiple_outputs():
    outputs = [{"desc": "Clean text"}, {"desc": "Naive Text Here Extra Stuff"}]
    html = ["<p>Clean text</p>", "<div>Naive Text Here <span>Extra Stuff</span></div>"]
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 1
    assert "'desc'" in feedback[0]


def test_validate_naive_match_whitespace_diff():
    """Tests that the validator catches naive matches even with different whitespace."""
    outputs = [
        {
            "desc": "Author  Name @handle\n· 5h   Main   content here",
        }
    ]
    html = ["<div><span>Author Name @handle · 5h</span><p>Main content here</p></div>"]
    # Naive text: "Author Name @handle · 5h Main content here"
    feedback = validate_naive_text_match(outputs, html)
    assert len(feedback) == 1
    assert "'desc'" in feedback[0]


# --- Tests for clean_agent_code ---


def test_clean_agent_code_raw_string():
    code = "def parse(): pass"
    assert clean_agent_code(code) == code


def test_clean_agent_code_markdown_python():
    code = "```python\ndef parse(): pass\n```"
    expected = "def parse(): pass"
    assert clean_agent_code(code) == expected


def test_clean_agent_code_markdown_simple():
    code = "```\ndef parse(): pass\n```"
    expected = "def parse(): pass"
    assert clean_agent_code(code) == expected


def test_clean_agent_code_json_dict_simple():
    code_dict = {"code": "def parse(): pass"}
    expected = "def parse(): pass"
    assert clean_agent_code(code_dict) == expected


def test_clean_agent_code_json_dict_other_key():
    code_dict = {"result": "import re\ndef parse(): pass"}
    expected = "import re\ndef parse(): pass"
    assert clean_agent_code(code_dict) == expected


def test_clean_agent_code_json_dict_no_code():
    code_dict = {"error": "failed"}
    expected = "{'error': 'failed'}"  # String representation of dict
    assert clean_agent_code(code_dict) == expected


def test_clean_agent_code_json_dict_code_with_markdown():
    code_dict = {"code": "```python\ndef parse(): pass\n```"}
    expected = "def parse(): pass"
    assert clean_agent_code(code_dict) == expected


def test_clean_agent_code_other_type():
    code_list = ["def parse(): pass"]
    expected = "['def parse(): pass']"  # String representation of list
    assert clean_agent_code(code_list) == expected


def test_clean_agent_code_none():
    assert clean_agent_code(None) == "None"


def test_clean_agent_code_empty_string():
    assert clean_agent_code("") == ""


def test_clean_agent_code_whitespace():
    code = "  \n def parse(): pass \n  "
    expected = "def parse(): pass"
    assert clean_agent_code(code) == expected


# --- Tests for validate_result ---


def test_validate_result_valid():
    obj = {
        "key_str": "value",
        "key_int": 123,
        "key_dict_str_str": {"a": "b", "c": "d"},
        "key_list_str": ["x", "y"],
        "key_list_mix": [
            "z",
            {"k1": "v1", "k2": 10, "k3": None},
            {"k4": "v4"},
        ],
        "key_list_empty_dict": [{}],
    }
    ok, msg = validate_result(obj)
    assert ok is True
    assert msg == "ok"


def test_validate_result_empty_dict():
    ok, msg = validate_result({})
    assert ok is False
    assert msg == "dict is empty"


def test_validate_result_not_a_dict():
    ok, msg = validate_result("string")
    assert ok is False
    assert msg == "result is not a dict"


def test_validate_result_non_string_key():
    ok, msg = validate_result({1: "value"})
    assert ok is False
    assert "non-string key" in msg


def test_validate_result_invalid_value_type():
    ok, msg = validate_result({"key": 1.23})  # float not allowed
    assert ok is False
    assert "invalid value" in msg
    assert "float" in msg


def test_validate_result_invalid_value_none():
    ok, msg = validate_result({"key": None})
    assert ok is False
    assert "invalid value" in msg
    assert "assigned None" in msg


def test_validate_result_invalid_dict_value():
    ok, msg = validate_result({"key": {"a": 1}})  # dict values must be str
    assert ok is False
    assert "invalid value" in msg
    assert "dict" in msg


def test_validate_result_invalid_list_item_type():
    ok, msg = validate_result({"key": ["a", 1.23]})  # float not allowed in list
    assert ok is False
    assert "invalid item at index 1" in msg
    assert "float" in msg
    assert "invalid internal types" in msg


def test_validate_result_invalid_list_item_dict_value():
    ok, msg = validate_result({"key": [{"a": 1.23}]})
    assert ok is False
    assert "invalid item at index 0" in msg
    assert "invalid internal types" in msg


def test_validate_result_invalid_list_item_dict_key():
    ok, msg = validate_result({"key": [{1: "a"}]})
    assert ok is False
    assert "invalid item at index 0" in msg
    assert "int" in msg
    assert "invalid internal types" in msg
